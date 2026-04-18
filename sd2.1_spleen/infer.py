"""Generate synthetic spleen ultrasound images from a text prompt using the
trained LoRA adapter.

Usage:
    python infer.py \\
        --lora-dir outputs/lora-spleen \\
        --prompt "an ultrasound image of a spleen" \\
        --num-images 8 \\
        --output-dir outputs/samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--lora-dir", required=True)
    p.add_argument("--prompt", default="an ultrasound image of a spleen")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/samples")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(args.lora_dir)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    remaining = args.num_images
    idx = 0
    while remaining > 0:
        bsz = min(args.batch_size, remaining)
        images = pipe(
            prompt=[args.prompt] * bsz,
            negative_prompt=[args.negative_prompt] * bsz if args.negative_prompt else None,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
            cross_attention_kwargs={"scale": args.lora_scale},
        ).images
        for im in images:
            im.save(out_dir / f"sample_{idx:04d}.png")
            idx += 1
        remaining -= bsz

    print(f"Saved {args.num_images} images to {out_dir}")


if __name__ == "__main__":
    main()
