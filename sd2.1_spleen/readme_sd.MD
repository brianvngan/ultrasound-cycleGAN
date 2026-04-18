# SD 2.1 Spleen Ultrasound LoRA (MVP)

Build an MVP pipeline to fine-tune Stable Diffusion 2.1 using LoRA on ~200 spleen ultrasound images, producing a model that generates synthetic spleen ultrasound images from a text prompt. Training runs on H200 GPU(s). No masks, no ControlNet — pure text-to-image for the MVP.

## Data notes

Raw images are 300×300, but SD 2.1 is trained at 512×512. We upscale:

- **Upscale to 512×512** (recommended) using Lanczos. Images are already square so no padding needed. SD 2.1's VAE expects 512×512, anything smaller gets resized internally anyway.

### Augmentations (on-the-fly, not baked into dataset)

- **Horizontal flip** — spleen ultrasound is symmetric enough that left-right flip produces valid images. 2× dataset for free, zero anatomical weirdness.
- **Small rotations (±5° to ±10°)** — mimics natural probe angle variation. Apply with black fill for corners (matches the ultrasound background). Values above ±15° start producing unrealistic orientations.

Implemented in [dataset.py](dataset.py) with `hflip_prob=0.5` and `max_rotation_deg=7.0` by default.

## Pipeline layout

| File | Role |
|------|------|
| [preprocess.py](preprocess.py) | 300→512 Lanczos upscale, writes `metadata.jsonl` captions |
| [dataset.py](dataset.py) | On-the-fly hflip + rotation, normalize to [-1, 1] |
| [train_lora.py](train_lora.py) | LoRA fine-tune of SD 2.1-base UNet (attention projections) |
| [infer.py](infer.py) | Load LoRA and sample images via DPM-Solver++ |
| [run.sh](run.sh) | End-to-end launcher |

---

## Running on CHIMERA24 (DGX H200)

All IMPACT students have access to **CHIMERA24** (the new DGX H200).

> **Returning user?** If you've already fine-tuned on chimera before, skip the one-time setup entirely and jump to [Per-session workflow](#per-session-workflow). You only need to install this project's `requirements.txt` into your existing env once.

---

### One-time setup (skip if already done)

The steps in this block are **per-account**, not per-project. If Miniconda is already on your home dir and you already have an `IMPACT` conda env from previous fine-tuning work, skip straight to the per-session workflow below.

<details>
<summary><b>1. Install Miniconda</b> — skip if <code>which conda</code> already returns a path</summary>

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Mini*
./Mini*
```
</details>

<details>
<summary><b>2. Create the IMPACT conda env</b> — skip if <code>conda env list</code> already shows <code>IMPACT</code></summary>

```bash
conda create -n IMPACT python=3.10 -y
conda activate IMPACT
pip install jupyterlab ipykernel
```
</details>

<details>
<summary><b>3. One-time LD_LIBRARY_PATH setup</b> — skip if <code>ls $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh</code> already exists</summary>

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Paste:

```bash
export LD_LIBRARY_PATH=$(python - <<'PY'
import site, os, glob
paths=[]
for p in site.getsitepackages():
    paths += glob.glob(os.path.join(p,"nvidia","*","lib"))
print(":".join(paths))
PY
):$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Reload:

```bash
conda deactivate
conda activate IMPACT
```
</details>

### Project dependencies (one-time per project)

Inside the `IMPACT` env, install this project's requirements:

```bash
conda activate IMPACT
pip install -r requirements.txt
```

---

### Per-session workflow

Do these every time you come back to train or sample.

#### 1. SSH into the headnode

```bash
ssh david.gogicajev001@chimera.umb.edu
```

**DO NOT RUN JOBS ON THE HEADNODE.** Allocate a compute node instead.

#### 2. Allocate an H200

**Quick test / sanity-check (60 min, 1 GPU):**

```bash
salloc -c2 -A impact -q aicore --gres=gpu:1 --mem=32G -w chimera24 -p AICORE_H200 -t 60
```

**Full training run (8 hours, 3 GPUs — the max this account can request):**

```bash
salloc -c6 -A impact -q aicore --gres=gpu:3 --mem=96G -w chimera24 -p AICORE_H200 -t 480
```

This grabs 6 CPU cores + 96 GB RAM + **3 H200 GPUs** (≈423 GB total VRAM) for 8 hours — enough headroom for full LoRA fine-tuning with large batches and long schedules.

#### 3. Activate the env and sanity-check the GPU

```bash
conda activate IMPACT
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `True NVIDIA H200`.

#### 4. Run the pipeline

From the project directory on the compute node:

```bash
# one-shot end-to-end (single GPU)
./run.sh

# or step-by-step, single GPU
python preprocess.py --input-dir raw_spleen_data --output-dir data/processed
accelerate launch train_lora.py --data-dir data/processed --output-dir outputs/lora-spleen --max-train-steps 3000 --train-batch-size 16
python infer.py --lora-dir outputs/lora-spleen --num-images 8 --output-dir outputs/samples
```

**Multi-GPU (3× H200 from the 8-hour allocation above):**

```bash
accelerate launch --num_processes=3 --multi_gpu train_lora.py \
    --data-dir data/processed \
    --output-dir outputs/lora-spleen \
    --train-batch-size 16 \
    --max-train-steps 3000
```

Each H200 has 141 GB VRAM, so a per-GPU `--train-batch-size 16` is comfortable for SD 2.1 LoRA at 512×512 (push to 24-32 if you want). Effective global batch = `num_processes × train_batch_size × gradient_accumulation_steps`, so `--num_processes=3 --train-batch-size 16` already gives you global batch 48 — you likely want to either **scale LR up** (≈√3×, so `--learning-rate 1.7e-4`) or **drop `--max-train-steps`** to ~1000 since each step now sees 3× the data.
