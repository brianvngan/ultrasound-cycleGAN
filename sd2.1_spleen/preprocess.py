"""Upscale raw 300x300 spleen ultrasound JPEGs to 512x512 (Lanczos) and
emit a metadata.jsonl caption file consumed by the training Dataset.

The raw images are already square, so no padding is needed (per readme).
Horizontal-flip and small-rotation augmentations are NOT baked in here —
they are applied on-the-fly inside dataset.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

TARGET_SIZE = 512
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def upscale_one(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.size != (TARGET_SIZE, TARGET_SIZE):
            im = im.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        im.save(dst, "PNG", compress_level=6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="raw_spleen_data")
    ap.add_argument("--output-dir", default="data/processed")
    ap.add_argument(
        "--caption",
        default="an ultrasound image of a spleen",
        help="Caption used for every image unless captions.jsonl is supplied.",
    )
    ap.add_argument(
        "--captions-file",
        default=None,
        help="Optional JSONL with {file_name, text} overriding --caption per image.",
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_file_caption: dict[str, str] = {}
    if args.captions_file:
        with open(args.captions_file) as f:
            for line in f:
                row = json.loads(line)
                per_file_caption[row["file_name"]] = row["text"]

    srcs = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in VALID_EXTS)
    if not srcs:
        raise SystemExit(f"No images found in {in_dir}")

    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w") as meta_f:
        for src in tqdm(srcs, desc="upscale 300->512"):
            dst_name = src.stem + ".png"
            dst = out_dir / dst_name
            upscale_one(src, dst)
            caption = per_file_caption.get(dst_name, args.caption)
            meta_f.write(json.dumps({"file_name": dst_name, "text": caption}) + "\n")

    print(f"Wrote {len(srcs)} images + metadata.jsonl to {out_dir}")


if __name__ == "__main__":
    main()
