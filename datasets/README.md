# datasets/

One folder per dataset, three splits, two authored subfolders plus two generated:

```
datasets/
├── <name>/
│   ├── train/
│   │   ├── im/          # 0001.jpg  (images)
│   │   ├── gt/          # 0001.png  (binary mask, 0 / 255)
│   │   ├── gt_edge/     # 0001.png  (clean edge line, the target)
│   │   └── gt_brush/    # 0001.png  (rough brush stroke, the 4th input channel)
│   ├── val/
│   │   ├── im/  gt/  gt_edge/  gt_brush/
│   ├── test/
│   │   ├── im/  gt/  gt_edge/  gt_brush/
│   └── meta.json        # written by preprocessing: counts and rejections
├── _raw/                # untouched sources (gitignored)
└── _qa/                 # contact sheets and reports (gitignored)
```

`train/`, `val/` and `test/` are exactly what you provide. There is no manifest
and no automatic splitting — if a near-duplicate sits in both `train/` and
`test/`, nothing will catch it for you.

## What the generated folders mean

| Folder | Content | Role |
|---|---|---|
| `gt_edge/` | the mask boundary drawn as a line, constant width, no jitter | what the model predicts |
| `gt_brush/` | the same boundary drawn as a rough stroke: simplified, wobbling, variable width | what the model receives |

Both cover the whole object — outer boundary and holes — so the input and the
target describe the same boundary. The brush is imprecise, not partial: it
wanders by up to 12 px and is several pixels wide, and only about 30% of its own
pixels land on the edge. `tools/verify_alignment.py` reports both numbers
(`edge-covered-by-brush`, `brush-on-edge`).

Both are derived from `gt/` only, never authored by hand. Either may be missing:
the trainer regenerates the brush for training samples anyway, and falls back to
generating it for validation when the file is absent.

## The alignment contract

For every sample:

```
im.shape[:2] == gt.shape[:2] == gt_edge.shape[:2] == gt_brush.shape[:2]
```

All four are produced from a single resize call during preprocessing, so they
cannot drift. Never resize one without the others.
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
To regenerate the two generated rasters on a prepared dataset without touching
`im/` or `gt/`:

```bash
python -m training.prepare.preprocess --dataset datasets/mything_prepared --inplace --force
```

## File naming

- Images and masks pair by **stem**, case-insensitively. Extensions may differ.
- `#`, spaces and unicode are allowed — DIS5K filenames contain `#`, and the
  media route URL-encodes it correctly.
- Avoid `/`, `\` and names starting with `.`.

## What is gitignored

`_raw/`, `_qa/`, and the pixel folders of every dataset (`im/`, `gt/`, `gt_edge/`,
`gt_brush/`). The pixels are reproducible from the source archive plus the config,
and committing them would bloat the repository and, for most datasets, violate
the licence.

Commit this file. If you ship a demo, commit the demo subset and its `meta.json`.
