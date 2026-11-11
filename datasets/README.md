# datasets/

Each dataset lives in its own folder with three parallel subfolders:

```
datasets/
├── <name>/
│   ├── rgb/          # 000000001.jpg   canonical short side 640
│   ├── gt/           # 000000001.png   binary mask, 0 / 255
│   ├── gt_edge/      # 000000001.png   synthetic user stroke, 0-255
│   └── meta.json     # import statistics and rejections
├── manifest.json     # index of every sample
├── _raw/             # downloads and converted sources (gitignored)
└── _qa/              # contact sheets and reports (gitignored)
```

## The alignment contract

For every sample:

```
rgb.shape[:2] == gt.shape[:2] == gt_edge.shape[:2]
```

All three are produced from a single resize call, so they cannot drift. Never
resize one without the other two. `tools/verify_alignment.py` enforces this and
writes a visual contact sheet.

## Canonical resolution

Images are resized so the short side is 640, **downscale only** — never upscaled.
Images already smaller than 640 keep their native size, which is compensated by
crop-time scale augmentation.

The reason for 640 rather than "match the smallest dataset": at 400 px short side
a 256 crop contains most of the object, so the model learns global shape. At
inference the crop contains a local boundary segment. That mismatch is fatal.

## manifest.json

The manifest is the single source of truth. Loaders must resolve paths through it,
never by string concatenation.

```json
{
  "version": 1,
  "datasets": ["demo"],
  "counts": {"demo": 12},
  "samples": [
    {
      "id": "demo/000000000",
      "dataset": "demo",
      "tier": "A",
      "rgb": "demo/rgb/000000000.jpg",
      "gt": "demo/gt/000000000.png",
      "gt_edge": "demo/gt_edge/000000000.png",
      "original_size": [900, 640],
      "canonical_size": [900, 640],
      "scale": 1.0,
      "split": "train",
      "instance_area_ratio": 0.114,
      "boundary_sharpness": 0.42,
      "quality_weight": 0.9,
      "license": "CC-BY-4.0-annotations/CC-BY-2.0-image",
      "stroke_seed": 0,
      "phash": "1234567890123456789"
    }
  ]
}
```

`split` is written by `training/prepare/split.py`, which assigns whole duplicate
groups (pHash, Hamming ≤ 6) to one split so near-duplicates cannot leak between
train and test.

## Adding a dataset

```bash
# 1. get images and masks into <source>/images and <source>/masks
# 2. normalise
python -m training.prepare.normalize --name mydata --source <source> --tier A

# 3. assign splits and verify
python -m training.prepare.split
python tools/verify_alignment.py
```

`--tier B` marks a research-only dataset. Checkpoints trained on tier B will not
load unless the config explicitly sets `data.tier: B`.

## What is gitignored

`_raw/`, `_qa/`, every `rgb/`, `gt/`, `gt_edge/` folder, and `manifest.json`. The
pixels are reproducible from the source archives plus the config; committing them
would bloat the repository and, for most datasets, violate the licence.

Commit `datasets/README.md` (this file) and, if you ship a demo, the demo subset
with its manifest.
