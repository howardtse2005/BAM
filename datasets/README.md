# datasets/

One folder per dataset, three splits, two authored subfolders plus one generated:

```
datasets/
├── <name>/
│   ├── train/
│   │   ├── im/          # 0001.jpg  (images)
│   │   ├── gt/          # 0001.png  (binary mask, 0 / 255)
│   │   └── gt_edge/     # 0001.png  (synthetic stroke, created by preprocessing)
│   ├── val/
│   │   ├── im/  gt/  gt_edge/
│   ├── test/
│   │   ├── im/  gt/  gt_edge/
│   └── meta.json        # written by preprocessing: counts and rejections
├── _raw/                # untouched sources (gitignored)
└── _qa/                 # contact sheets and reports (gitignored)
```

`train/`, `val/` and `test/` are exactly what you provide. There is no manifest
and no automatic splitting — if a near-duplicate sits in both `train/` and
`test/`, nothing will catch it for you.

## The alignment contract

For every sample:

```
im.shape[:2] == gt.shape[:2] == gt_edge.shape[:2]
```

All three are produced from a single resize call during preprocessing, so they
cannot drift. Never resize one without the other two.
`tools/verify_alignment.py` enforces this and writes a visual contact sheet.

## Canonical resolution

Preprocessing downscales so the short side is 640, **never upscaling**. Images
already smaller than 640 keep their native size; crop-time scale augmentation
compensates.

The reason for 640 rather than "match the smallest dataset": at 400 px short side
a 256 crop contains most of the object, so the model learns global shape. At
inference the crop contains a local boundary segment. That mismatch is fatal.

## Adding a dataset

```bash
# 1. lay it out
datasets/mything/train/im/0001.jpg
datasets/mything/train/gt/0001.png
datasets/mything/val/...    datasets/mything/test/...

# 2. preprocess (writes a sibling folder by default)
python -m training.prepare.preprocess --dataset datasets/mything
# -> datasets/mything_prepared

# 3. verify
python tools/verify_alignment.py --dataset datasets/mything_prepared

# 4. point config.yaml at it
#    dataset: mything_prepared
```

Preprocessing is non-destructive by default. `--inplace` rewrites the source
folder and prints a warning; use `--backup` to keep a copy under `_original/`.

## File naming

- Images and masks pair by **stem**, case-insensitively. Extensions may differ.
- `#`, spaces and unicode are allowed — DIS5K filenames contain `#`, and the
  media route URL-encodes it correctly.
- Avoid `/`, `\` and names starting with `.`.

## What is gitignored

`_raw/`, `_qa/`, and the pixel folders of every dataset. The pixels are
reproducible from the source archive plus the config, and committing them would
bloat the repository and, for most datasets, violate the licence.

Commit this file. If you ship a demo, commit the demo subset and its `meta.json`.
