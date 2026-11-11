# BAM — Brush Anything Model

Stroke-guided edge segmentation for any object. You draw a rough line along the boundary; a small 4-channel UNet (RGB + stroke) snaps it to the true edge, and a deterministic post-process turns that edge into a filled mask.

Runs on CPU. No GPU required at inference.

```
stroke → 256×256 crops → UNet (0.34M params) → merge → edge → fill → mask
```

---

## The pipeline in four steps

| # | Step | Command | Produces |
|---|---|---|---|
| 1 | Create `gt_edge/` | (same command as step 2) | a synthetic user stroke per mask |
| 2 | Preprocess | `python -m training.prepare.preprocess --dataset datasets/<name>` | downscaled `im/`, `gt/`, `gt_edge/` |
| 3 | Train and validate | `python -m training.train` | `model/bam_v1.pth` |
| 4 | Deploy | `python run.py` | the web app on `:8000` |

Steps 1 and 2 are one command because the stroke has to be generated at the final
resolution — you cannot create `gt_edge/` before the mask has been downscaled.
They are described separately below because they produce different artifacts and
have different knobs.

---

## Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Dataset layout

Every dataset is a folder with three splits. Each split has `im/` (images) and
`gt/` (binary masks). `gt_edge/` is generated, never authored by hand.

```
datasets/
└── <name>/
    ├── train/
    │   ├── im/        0001.jpg  0002.jpg ...
    │   ├── gt/        0001.png  0002.png ...
    │   └── gt_edge/   0001.png  0002.png ...   <- created by step 1
    ├── val/
    │   ├── im/  gt/  gt_edge/
    ├── test/
    │   ├── im/  gt/  gt_edge/
    └── meta.json                                <- written by step 2
```

Rules:

- Images and masks are paired by **filename stem**, case-insensitively. Any
  common extension works (`.jpg`, `.png`, ...). Names may contain `#`, spaces and
  unicode — DIS5K filenames do.
- The split is whatever you put in `train/`, `val/` and `test/`. There is no
  manifest and no automatic splitting.
- Masks must be single-channel, with the object at 255 and the background at 0.

---

## Step 1 — Create `gt_edge/`

`gt_edge/` is the **synthetic user stroke**: a raster that looks like a person
dragging a brush roughly along the boundary. It is the model's 4th input channel,
and it is what makes the task learnable — without it the network would have to
find the boundary from scratch.

It is generated from each mask by `training/stroke_synth.py`:

1. **Contour extraction** — ordered contours from `cv2.findContours`, holes included.
2. **Simplification** — RDP with `stroke_simplify` epsilon (2–10 px). People cut corners.
3. **Smoothing** — a Hann window of `stroke_smooth` (5–21 px).
4. **Deformation** — displacement along the local normal: a global offset
   (`stroke_jitter`, σ≈3 px), two sinusoids (`stroke_wobble_amp`,
   `stroke_wobble_len`), and fractal noise. Clipped to `stroke_max_offset` (12 px).
5. **Coverage** — 1–3 arcs covering `stroke_coverage` (45–100%) of the perimeter.
   15% of samples get only a short scribble.
6. **Rasterisation** — anti-aliased brush of radius `stroke_radius_min..max`
   (1.5–6 px), with occasional dry-brush gaps.
7. **Mistakes** — 15% distractor stroke along a strong non-object gradient, 5%
   interior scribble, 8% no stroke at all.

Where it is used:

| Split | Stroke source | Why |
|---|---|---|
| `train` | regenerated every sample | unbounded augmentation; the model never sees the same stroke twice |
| `val`, `test` | the stored `gt_edge/*.png` | evaluation must be reproducible |

To inspect what the model will actually see:

```bash
python tools/visualize_crops.py --split train --count 6
# writes datasets/_qa/crops/crops_train.jpg
```

---

## Step 2 — Preprocess (downscale, validate, create `gt_edge/`)

```bash
python -m training.prepare.preprocess --dataset datasets/DIS5K
```

By default the output is a **sibling** folder, so a 4K source is never modified:

```
datasets/DIS5K            (source, untouched)
datasets/DIS5K_prepared   (output: canonical size + gt_edge)
```

Then point the config at it:

```yaml
dataset: DIS5K_prepared
```

What the command does, per split:

1. **Pairs** `im/` and `gt/` by stem, skipping images with no mask.
2. **Validates** — rejects empty masks (<0.2% foreground), masks that cover almost
   everything (>98%), and degenerate boundaries. Rejections are listed in `meta.json`.
3. **Downscales** image and mask so the short side is `short_side` (640). Both are
   resized from the same dimensions, so they cannot drift apart. Images already
   smaller than 640 are left alone — never upscaled.
4. **Writes `gt_edge/`** at the final resolution, seed 0.
5. **Writes `meta.json`** with per-split counts and rejection reasons.

**Cropping is not a preprocessing step.** The 256×256 crop is sampled on the fly
during training, and the sampler deliberately varies position, scale and content
(boundary-centred, near-boundary, empty, cluttered). A 640×640 image would yield
dozens of overlapping crops; materialising them would multiply disk usage and
destroy that variation.

Useful flags:

```bash
--out datasets/other_name   # explicit output folder
--inplace                   # rewrite the dataset itself (destructive, warns)
--backup                    # copy im/gt to _original/ before an in-place run
--dry-run                   # report only
--limit 500                 # first N images per split
--force                     # regenerate existing gt_edge
--no-edge                   # skip stroke generation
```

### Verify before training

```bash
python tools/verify_alignment.py --dataset datasets/DIS5K_prepared
```

```
train  3000 images | shape 0 | non-binary 0 | oversized 0 | no-edge 0 | stroke-near-boundary 0.968
val     470 images | shape 0 | non-binary 0 | oversized 0 | no-edge 0 | stroke-near-boundary 0.931
test   2000 images | shape 0 | non-binary 0 | oversized 0 | no-edge 0 | stroke-near-boundary 0.980
```

`shape` and `non-binary` must be 0. `oversized` means the split was not
preprocessed. `stroke-near-boundary` is the fraction of stroke pixels within
20 px of the mask boundary — below ~0.75 means the generator is misconfigured.

A contact sheet lands in `datasets/_qa/alignment/alignment_<split>.jpg`.

---

## Step 3 — Train and validate

Set the dataset in `config.yaml`:

```yaml
dataset: DIS5K_prepared
width: 20            # 16 = faster, 32 = more accurate
batch_size: 32
epochs: 60
```

```bash
python -m training.train --dry-run     # one synthetic batch, no data needed
python -m training.train               # full run
```

The loop: AdamW, cosine schedule with warmup, AMP on CUDA, EMA weights, early
stopping on validation boundary F@2px. Best checkpoint is copied to
`model/bam_v1.pth`; everything else goes to `runs/<timestamp>/`.

Each epoch reports:

```
[epoch 12] loss 0.1842 val_bf 0.8013 val_iou 0.8641 val_biou 0.6420 (94s)
```

`val_bf` is boundary F-measure with a 2 px tolerance — the metric that matters,
because the product needs the edge in the right place, not merely a plausible mask.

Validation crops are always boundary-centred. Training crops use the four-way
mixture in `config.yaml` (`mix_boundary`, `mix_near`, `mix_empty`, `mix_dense`);
an empty crop teaches the model to stay silent, but it would give free 1.0 scores
in validation.

Evaluate on the held-out split:

```bash
python -m training.evaluate --split test
```

```
| metric | value |
| --- | --- |
| iou | 0.8612 |
| boundary_iou | 0.6403 |
| bf | 0.8044 |
| mae | 0.0210 |
```

Export for CPU inference (optional, 2–4× faster):

```bash
python -m training.export_onnx --checkpoint model/bam_v1.pth --int8
```

The backend prefers `onnx_int8` → `onnx_fp32` → `torch` → `mock`.

---

## Step 4 — Deploy the website

```bash
python run.py
# [bam] serving on http://127.0.0.1:8000
```

```bash
python run.py --host 0.0.0.0 --serve waitress    # LAN / production
```

What you get:

1. **Dataset picker** — every folder under `datasets/` with split counts and a thumbnail.
2. **Editor** — left tool rail, centre canvas, right inspector, status bar.
   - `B` brush, `E` eraser, `M` magic pen, `F` fill holes, `Space` pan
   - `[` / `]` brush size, `+` / `-` zoom, `0` fit, `Tab` mask, `Shift+Tab` edge
   - split selector in the top bar (`train` / `val` / `test`)
3. **Magic pen** — draw a rough loop, release, get a filled mask. `Alt`+drag subtracts.
4. **Save** — writes `<dataset>/<split>/gt/<name>.png` at the image's native size.

The canvas always works at the canonical short side (640), even when the stored
image is 4K. That keeps the object scale identical to training and bounds CPU cost
regardless of the source resolution.

**It starts without a checkpoint.** With no `model/bam_v1.pth`, the registry falls
back to a mock backend that predicts a band around your stroke, so the whole path
can be exercised. `/api/health` reports `model: mock` and the UI shows an amber
warning. Nothing is silently faked.

### API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | status, backend, device, data tier |
| `GET` | `/api/datasets` | datasets with per-split counts |
| `GET` | `/api/datasets/<ds>/images?split=train` | image names |
| `GET` | `/media/<ds>/<split>/<im\|gt\|gt_edge>/<name>` | image bytes |
| `POST` | `/api/segment` | crops + stroke → mask + edge |
| `POST` | `/api/mask/save` | save a mask |
| `POST` | `/api/image/upload` | upload and canonicalise |
| `POST` | `/magic_pen/predict_crops` | masker-compatible, returns an edge mask |

`/api/segment` takes crop **boxes**, not crop images; the server slices them from
the stored image with replicate padding. That removes a class of base64 alignment
bugs and cuts the payload roughly 20×.

---

## Configuration

One flat list, one setting per line. See `config.py` for all of them.

```yaml
dataset: demo                 # folder under datasets_dir
datasets_dir: ./datasets
short_side: 640               # downscale target
crop_size: 256
width: 20                     # UNet base channels: 16 / 20 / 24 / 32
depth: 4
batch_size: 32
epochs: 60
lr: 0.0003
train_device: auto
infer_device: cpu
backend: auto
```

Override without editing the file:

```bash
BAM_BATCH_SIZE=16 python run.py
python run.py --set port=8001 --set width=24
```

Resolution order: CLI > environment > `config.yaml` > defaults.

---

## How the model works

**Input** `4 × 256 × 256`: RGB plus the stroke channel.
**Output** `1 × 256 × 256`: a soft boundary band.

The target is a Gaussian-blurred boundary (σ = `target_sigma`), not a 1-px line —
trainable under class imbalance and tolerant of 1-px annotation noise. Loss is
`0.5·BCE(pos_weight=6) + 0.5·Dice + 0.1·L1`, masked by a validity map so padded
crop regions contribute nothing.

The architecture is a UNet with depthwise-separable blocks, width 20, depth 4:
**0.34 M parameters, 1.58 GFLOPs** at 256². Measured on a 16-core CPU with
`channels_last`: **18 ms/crop** at batch 1, 24 ms at batch 8.

**Merging** preserves the reference `PredictionMerger`: average overlapping crop
predictions, boost repeatedly brushed pixels, threshold, denoise, morphology.

**Edge → mask** is deterministic:

1. Threshold and thin the edge (Zhang-Suen).
2. Dilate into a 3-px barrier so the fill cannot leak through 1-px gaps.
3. Label free space; the object is the enclosed component **adjacent to the stroke
   ring**. Holes are separate components and are therefore preserved.
4. Refine with a guided filter; fall back to GrabCut, then to the raw loop.
5. An open stroke cannot define a region, so the endpoint returns the predicted
   edge and flags `mode_used: "open_loop"` rather than guessing a blob.

---

## Repository layout

```
BAM/
├── datasets/            # <name>/{train,val,test}/{im,gt,gt_edge} + meta.json
├── model/               # trained weights (bam_v1.pth / .onnx)
├── frontend/
│   ├── templates/       # base, select_dataset, index
│   └── static/{css,js}
├── backend/
│   ├── app.py           # Flask factory
│   ├── routes/          # pages, datasets, masks, magic_pen, health
│   ├── inference/       # predictor, merger, edge_to_mask, registry
│   └── services/        # storage, dataset_index, cache
├── training/
│   ├── prepare/         # download, preprocess, dedupe
│   ├── unet.py          # BAMUNet, self-contained
│   ├── paths.py         # split scanning, torch-free
│   ├── stroke_synth.py  # the data IP
│   ├── dataset.py, sampler.py, losses.py, metrics.py
│   └── train.py, evaluate.py, export_onnx.py
├── tools/               # make_demo_subset, verify_alignment, visualize_crops, benchmark_cpu
├── tests/               # 103 tests
├── config.py            # flat dataclass config
├── config.yaml
├── imgio.py             # unicode-safe image IO
└── run.py
```

`model/` holds weights only. The architecture lives in `training/unet.py` and
imports nothing from `config`, so it can be loaded standalone. The backend imports
it lazily, so the web app still starts without PyTorch.

---

## Demo without a real dataset

```bash
python tools/make_demo_subset.py --count 24
python -m training.prepare.preprocess --dataset datasets/_raw/demo --out datasets/demo
python -m training.train --set epochs=1
python run.py
```

Synthetic shapes, not training data — enough to exercise every stage.

---

## Licensing

**This matters.** Two commonly cited saliency datasets are not safe for commercial use:

| Dataset | Status |
|---|---|
| **Open Images V7** | images CC BY 2.0, masks CC BY 4.0 — commercial use with attribution |
| **COCO 2017** | annotations CC BY 4.0; images under Flickr terms, verify per image |
| **DIS5K** | no explicit licence — all rights reserved by default |
| **DUTS** | *"All rights reserved by the original authors"* |
| **ECSSD / HKU-IS / DUT-OMRON** | research datasets, no commercial grant |
| **SA-1B** | SA-1B Dataset Research License |
| **Cityscapes / Mapillary / BDD100K / Objects365 / xView** | non-commercial or research |

Use `tier: A` for anything you ship and `tier: B` for research. Every checkpoint
records its tier, and the backend refuses to load a tier-B checkpoint unless the
config explicitly says `tier: B`.

Attribution is a legal obligation. If you redistribute the app, keep these credits.

---

## Limitations

- **Thin structures** (wires, cracks): use `mode: "line"`, which returns an edge
  ribbon instead of a filled region.
- **Transparent or heavily occluded objects**: boundaries are genuinely ambiguous.
- **Objects under ~16 px** at the working resolution: below the model's useful scale.
- **Open strokes** cannot be filled; the UI tells the user to close the loop.
- **One object per loop**: multiple loops in one pass are unioned.

---

## Testing

```bash
pytest -q
python -m pyflakes backend training tools tests config.py imgio.py run.py
```

103 tests cover config precedence, stroke-synthesis invariants, joint-transform
alignment, crop-bucket coverage, model shapes and checkpoint round-trips, merge
arithmetic, the fill algorithm (hole preservation, open-loop handling), the API
contract across splits, and path-traversal rejection.
