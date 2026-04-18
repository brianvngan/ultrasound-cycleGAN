"""LoRA fine-tuning for Stable Diffusion 2.1 (base, 512x512) on spleen ultrasound.

Usage (single H200):
    accelerate launch train_lora.py \\
        --data-dir data/processed \\
        --output-dir outputs/lora-spleen \\
        --max-train-steps 3000

Optimizations applied:
  * Text embeddings for every unique caption are computed ONCE at startup and
    cached; the text encoder is then moved to CPU (the MVP dataset has a single
    caption, so this cuts out a text-encoder forward every step).
  * TF32 matmul + cuDNN benchmark enabled (Hopper).
  * Optional `torch.compile(unet)` — on by default, disable with --no-compile.

Trained adapter is saved via the standard diffusers/peft LoRA format and can
be loaded with `pipe.load_lora_weights(output_dir)`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import SpleenUltrasoundDataset, collate_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output-dir", default="outputs/lora-spleen")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-train-steps", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--lr-warmup-steps", type=int, default=100)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rotation-deg", type=float, default=7.0)
    p.add_argument("--hflip-prob", type=float, default=0.5)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the UNet (~20-30%% speedup on H200).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)

    train_dataset = SpleenUltrasoundDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        hflip_prob=args.hflip_prob,
        max_rotation_deg=args.max_rotation_deg,
    )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.eval()
    with torch.no_grad():
        tok = tokenizer(
            train_dataset.unique_captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(accelerator.device)
        cached_text_embeds = text_encoder(tok)[0].detach()
    del text_encoder
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        print(
            f"Cached text embeds for {len(train_dataset.unique_captions)} unique caption(s); "
            f"text encoder released."
        )

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=1e-2)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )

    if args.compile:
        if accelerator.is_main_process:
            print("torch.compile(unet) — first step will be slow while compiling.")
        unet = torch.compile(unet, mode="default", fullgraph=False)

    if accelerator.is_main_process:
        print(
            f"dataset={len(train_dataset)}  batch={args.train_batch_size}  "
            f"steps/epoch={steps_per_epoch}  epochs~={num_epochs}  "
            f"max_steps={args.max_train_steps}"
        )

    global_step = 0
    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    unet.train()
    done = False
    for epoch in range(num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = cached_text_embeds[batch["caption_idx"]]

                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=float(loss.detach().item()), lr=lr_scheduler.get_last_lr()[0])

                if global_step % args.checkpoint_every == 0 and accelerator.is_main_process:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    _save_lora(accelerator.unwrap_model(unet), ckpt_dir)

                if global_step >= args.max_train_steps:
                    done = True
                    break
        if done:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_lora(accelerator.unwrap_model(unet), output_dir)
        print(f"Saved LoRA adapter to {output_dir}")

    accelerator.end_training()


def _save_lora(unet, out_dir: Path) -> None:
    inner = getattr(unet, "_orig_mod", unet)
    state_dict = get_peft_model_state_dict(inner)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(out_dir),
        unet_lora_layers=state_dict,
        safe_serialization=True,
    )


if __name__ == "__main__":
    main()
