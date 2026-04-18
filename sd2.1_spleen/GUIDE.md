# Step-by-Step Guide — SD 2.1 Spleen LoRA on CHIMERA24

Linear walkthrough from a clean terminal on your laptop to generated sample images. Steps already covered in previous fine-tuning work on chimera are marked **[skip if already done]** — run the check command first and skip the block if the check succeeds.

---

## 0. Prereqs on your laptop

- SSH access to `chimera.umb.edu` as `david.gogicajev001`.
- This project folder (`sd2.1_spleen/`) on your laptop, including `raw_spleen_data/`.

---

## 1. Copy the project to chimera

From your laptop:

```bash
cd /Users/davidgogi/Desktop
rsync -avz --progress \
    --exclude='.DS_Store' \
    --exclude='__pycache__' \
    --exclude='.claude' \
    --exclude='outputs' \
    --exclude='data/processed' \
    sd2.1_spleen/ david.gogicajev001@chimera.umb.edu:~/sd2.1_spleen/
```

*(`rsync` skips unchanged files on re-runs, so this is safe to repeat after edits. The excludes keep junk files and remote-only artifacts — checkpoints, preprocessed data — from being clobbered by uploads.)*

---

## 2. SSH into the chimera headnode

```bash
ssh david.gogicajev001@chimera.umb.edu
```

**DO NOT RUN JOBS ON THE HEADNODE.** It's just a launch pad.

---

## 3. Install Miniconda — [skip if already done]

**Check first:**

```bash
conda --version
```

If that prints a version (e.g. `conda 24.x.x`), **skip this step**. Your prompt showing `(base)` is also a dead giveaway that conda is already active.

*(Note: `which conda` prints a shell function body instead of a path — that's normal, not a problem. Use `conda --version` for an unambiguous check.)*

Otherwise:

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Mini*
./Mini*
```

Accept the license, accept the default install path, and say **yes** when it asks to run `conda init`. Then:

```bash
source ~/.bashrc
```

---

## 4. Create the IMPACT conda env — [skip if already done]

**Check first:**

```bash
conda env list | grep IMPACT
```

If that prints a line, **skip this step**. Otherwise:

```bash
conda create -n IMPACT python=3.10 -y
conda activate IMPACT
pip install jupyterlab ipykernel
```

---

## 5. One-time LD_LIBRARY_PATH setup — [skip if already done]

**Check first:**

```bash
conda activate IMPACT
ls $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh 2>/dev/null
```

If that prints a path, **skip this step**. Otherwise:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Paste exactly this, then save (Ctrl+O, Enter, Ctrl+X):

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

---

## 6. Install project requirements

Do this once per project (or whenever `requirements.txt` changes). **Install PyTorch separately with the CUDA build that matches the cluster driver** — on CHIMERA24 that's CUDA 12.8 — then install the rest:

```bash
conda activate IMPACT
cd ~/sd2.1_spleen

# 1) PyTorch built for CUDA 12.8 (cluster driver supports up to CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) Everything else
pip install -r requirements.txt
```

> **Why the split?** Plain `pip install torch` grabs the newest wheel from PyPI, which may be built against CUDA 13.x — the cluster driver can't load those, and you'll see a misleading *"NVIDIA driver on your system is too old (found version 12080)"* error. The `--index-url` above pins the CUDA-12.8 build.

---

## 7. Allocate an H200 node (per session)

From the **headnode** (this is the big allocation — 3 GPUs, 96 GB RAM, 8 hours):

```bash
salloc -c6 -A impact -q aicore --gres=gpu:3 --mem=96G -w chimera24 -p AICORE_H200 -t 480
```

You'll land in a new shell on `chimera24` once the scheduler grants the job. Your prompt should change.

*(For a quick 60-min smoke test, use `salloc -c2 -A impact -q aicore --gres=gpu:1 --mem=32G -w chimera24 -p AICORE_H200 -t 60` instead.)*

---

## 8. Re-activate env on the compute node

`salloc` drops you into a fresh shell on the compute node — reactivate the env:

```bash
conda activate IMPACT
cd ~/sd2.1_spleen
```

---

## 9. GPU sanity check

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Expected output:

```
True 3 NVIDIA H200
```

If `torch.cuda.is_available()` is `False`, the `LD_LIBRARY_PATH` setup from step 5 likely didn't reload — re-run `conda deactivate && conda activate IMPACT`.

**If you see `NVIDIA driver on your system is too old (found version 12080)`:** your PyTorch was built against CUDA 13.x but the cluster driver only supports CUDA 12.8. Reinstall with the matching wheels:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then retry the sanity check.

---

## 10. Preprocess the dataset (300 → 512 PNG)

```bash
python preprocess.py --input-dir raw_spleen_data --output-dir data/processed
```

Takes a few seconds. Produces `data/processed/*.png` and `data/processed/metadata.jsonl`.

---

## 11. Train the LoRA (3× H200, multi-GPU)

```bash
accelerate launch --num_processes=3 --multi_gpu train_lora.py \
    --data-dir data/processed \
    --output-dir outputs/lora-spleen \
    --train-batch-size 16 \
    --max-train-steps 1000 \
    --learning-rate 1.7e-4
```

Notes:
- Global batch = `3 × 16 = 48`, so `--max-train-steps 1000` already sees a lot of data. Bump to 2000-3000 if samples still look off.
- LR is scaled ≈√3× vs. the single-GPU default (`1e-4`) to match the larger global batch.
- First step is slow because `torch.compile` is warming up; ignore the initial pause.
- Checkpoints land in `outputs/lora-spleen/checkpoint-<step>/` every 1000 steps; the final adapter goes in `outputs/lora-spleen/`.

**Single-GPU variant** (drop to this if you used the 1-GPU salloc):

```bash
accelerate launch train_lora.py \
    --data-dir data/processed \
    --output-dir outputs/lora-spleen \
    --train-batch-size 16 \
    --max-train-steps 3000
```

---

## 12. Generate sample images

```bash
python infer.py \
    --lora-dir outputs/lora-spleen \
    --prompt "an ultrasound image of a spleen" \
    --num-images 8 \
    --output-dir outputs/samples
```

Outputs land in `outputs/samples/sample_0000.png` … `sample_0007.png`.

---

## 13. Copy results back to your laptop

From your laptop (open a second terminal so you don't kill the salloc):

```bash
rsync -avz --progress david.gogicajev001@chimera.umb.edu:~/sd2.1_spleen/outputs/ /Users/davidgogi/Desktop/sd2.1_spleen/outputs/
```

---

## 14. Release the GPU when done

Back in the compute-node shell:

```bash
exit
```

This drops you back to the headnode and frees the allocation. **Always exit when you're finished** — idle salloc sessions eat your quota.

Then one more `exit` to log out of chimera entirely.
