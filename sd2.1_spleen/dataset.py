"""Dataset for SD 2.1 LoRA fine-tuning on spleen ultrasound images.

On-the-fly augmentations (per readme):
  * Random horizontal flip (p=0.5) — spleen ultrasound is L/R symmetric.
  * Random small rotation in [-max_rotation_deg, +max_rotation_deg] with black fill,
    mimicking probe-angle variation.

Images are expected to already be 512x512 (run preprocess.py first).
Output tensors are normalized to [-1, 1] as expected by the SD VAE encoder.

The dataset exposes `unique_captions` + returns a `caption_idx` per example so
the training loop can pre-compute text-encoder outputs once and look them up
by index (avoids running the text encoder every step).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class SpleenUltrasoundDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        resolution: int = 512,
        hflip_prob: float = 0.5,
        max_rotation_deg: float = 7.0,
        cache_in_memory: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found — run preprocess.py first")

        entries: list[dict] = []
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        caption_to_idx: dict[str, int] = {}
        self.caption_indices: list[int] = []
        self.unique_captions: list[str] = []
        for e in entries:
            cap = e["text"]
            if cap not in caption_to_idx:
                caption_to_idx[cap] = len(self.unique_captions)
                self.unique_captions.append(cap)
            self.caption_indices.append(caption_to_idx[cap])

        self.entries = entries
        self.resolution = resolution
        self.hflip_prob = hflip_prob
        self.max_rotation_deg = max_rotation_deg

        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self._cache: list[Image.Image | None] = [None] * len(entries)
        if cache_in_memory:
            for i, e in enumerate(entries):
                img = Image.open(self.data_dir / e["file_name"]).convert("RGB")
                if img.size != (resolution, resolution):
                    img = img.resize((resolution, resolution), Image.LANCZOS)
                img.load()
                self._cache[i] = img

    def __len__(self) -> int:
        return len(self.entries)

    def _augment(self, img: Image.Image) -> Image.Image:
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)

        if random.random() < self.hflip_prob:
            img = TF.hflip(img)

        if self.max_rotation_deg > 0:
            angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)

        return img

    def __getitem__(self, idx: int) -> dict:
        img = self._cache[idx]
        if img is None:
            entry = self.entries[idx]
            img = Image.open(self.data_dir / entry["file_name"]).convert("RGB")

        img = self._augment(img)
        pixel_values = self.normalize(img)
        return {
            "pixel_values": pixel_values,
            "caption_idx": torch.tensor(self.caption_indices[idx], dtype=torch.long),
        }


def collate_fn(examples: list[dict]) -> dict:
    pixel_values = torch.stack([e["pixel_values"] for e in examples]).to(
        memory_format=torch.contiguous_format
    ).float()
    caption_idx = torch.stack([e["caption_idx"] for e in examples])
    return {"pixel_values": pixel_values, "caption_idx": caption_idx}
