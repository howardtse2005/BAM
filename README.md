# BAM — Brush Anything Model

Brush-guided edge refinement for any object. You drag a rough stroke along a
boundary; a small 4-channel UNet (RGB + brush) snaps it onto the true edge and
returns a clean line; a deterministic fill turns that line into a mask.

Runs on CPU. No GPU required at inference.

```
brush → stroke square → UNet (0.34 M params) → clean line → fill → mask
```

---

## The pipeline in four steps

| # | Step | Command | Produces |
|---|---|---|---|
| 1 | Generate the rasters | (same command as step 2) | `gt_edge/` and `gt_brush/` per mask |
| 2 | Preprocess | `python -m training.prepare.preprocess --dataset datasets/<name>` | downscaled `im/`, `gt/`, `gt_edge/`, `gt_brush/` |
| 3 | Train and validate | `python -m training.train` | `model/bam_v1.pth` |
| 4 | Deploy | `python run.py` | the web app on `:8000` |

Steps 1 and 2 are one command because both rasters have to be generated at the
final resolution — a stroke drawn at 4K and then shrunk loses exactly the
sub-pixel wobble the model is supposed to correct.

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

Every dataset is a folder with three splits. Each split has `im/` (images),
`gt/` (binary masks), and two generated rasters: `gt_edge/` and `gt_brush/`.

```
datasets/
└── <name>/
    ├── train/
    │   ├── im/        0001.jpg  0002.jpg ...
    │   ├── gt/        0001.png  0002.png ...
    │   ├── gt_edge/   0001.png  0002.png ...   <- the clean line to predict
    │   └── gt_brush/  0001.png  0002.png ...   <- the rough stroke to consume
    ├── val/
    │   ├── im/  gt/  gt_edge/  gt_brush/
    ├── test/
    │   ├── im/  gt/  gt_edge/  gt_brush/
    └── meta.json                                <- written by step 2
```

Rules:

- Images and masks are paired by **filename stem**, case-insensitively. Any
  common extension works (`.jpg`, `.png`, ...). Names may contain `#`, spaces and
  unicode — DIS5K filenames do.
- The split is whatever you put in `train/`, `val/` and `test/`. There is no
  manifest and no automatic splitting.
- Masks must be single-channel, with the object at 255 and the background at 0.
- `gt_edge/` and `gt_brush/` are never authored by hand. Either may be missing;
  the preprocessing step regenerates what it needs from `gt/`.

---

## The two rasters

Both are produced from the same mask by `training/stroke_synth.py`, from the same
contour walk and the same rasteriser. The difference is what is removed.

### `gt_edge` — the clean line

The target. Full perimeter coverage, no simplification, no smoothing, no
normal-direction displacement, constant radius. Deterministic: the same mask
always yields the same raster, which is what makes it usable as a target.

```python
from training.stroke_synth import clean_edge
edge = clean_edge(mask)        # uint8, 0/255
```

### `gt_brush` — the rough stroke

The input. What a person dragging a brush along a boundary produces. Every
contour is stroked end to end, so the brush covers the same boundary as
`gt_edge` — it is just imprecise:

1. **Contour extraction** — ordered contours from `cv2.findContours`, holes included.
2. **Simplification** — RDP with `brush_simplify` epsilon (2–10 px). People cut corners.
3. **Smoothing** — a Hann window of `brush_smooth` (5–21 px).
4. **Deformation** — displacement along the local normal: a global offset
   (`brush_jitter`, σ≈3 px), two sinusoids (`brush_wobble_amp`,
   `brush_wobble_len`), and fractal noise. Clipped to `brush_max_offset` (12 px).
5. **Coverage** — the full perimeter, `brush_coverage` (100%). Every contour gets
   its own ring, so holes are outlined too.
6. **Rasterisation** — anti-aliased brush of radius `brush_radius_min..max`
   (1.5–6 px), so the stroke is several pixels wide and its width varies.

```python
from training.stroke_synth import synthesize_brush
brush = synthesize_brush(mask, rgb=rgb, rng=0)   # uint8, 0/255
```

Coverage is deliberately not partial. A brush that stopped halfway would leave
the 4th channel describing one part of the object and the target describing all
of it, and the model would be asked to invent the rest. `verify_alignment`
measures this directly: **`edge-covered-by-brush` ≈ 1.0** means every pixel of
the target has brush nearby.

The stroke is still rough enough to be a real task: it is displaced by up to
12 px, its radius varies from 1.5 to 6 px, and only ~30% of its own pixels land
on the edge (**`brush-on-edge`**). Copying the input channel therefore scores
badly.

The knobs that *would* break alignment — `brush_coverage` below 1, `p_partial`,
`p_two_arcs`, `p_no_brush`, `p_distractor`, `p_interior`, `p_dry_brush` — all
default to off. Set them if you want to train a model that also copes with
half-drawn strokes.

### Where each one is used

| Split | Brush source | Why |
|---|---|---|
| `train` | regenerated every sample | unbounded augmentation; the model never sees the same stroke twice |
| `val`, `test` | the stored `gt_brush/*.png` | evaluation must be reproducible |

The edge target is always rebuilt from the crop's own mask, so it stays
pixel-aligned through flips, rotations and rescaling. `tools/verify_alignment.py`
checks that the stored `gt_edge/` agrees with it.

To inspect what the model will actually see:

```bash
python tools/visualize_crops.py --split train --count 6
# writes datasets/_qa/crops/crops_train.jpg
```

Each row is `[ rgb crop ] [ brush channel ] [ edge target over rgb ]`.

---

## Step 2 — Preprocess (downscale, validate, generate both rasters)

```bash
python -m training.prepare.preprocess --dataset datasets/DIS5K
```

By default the output is a **sibling** folder, so a 4K source is never modified:

```
datasets/DIS5K            (source, untouched)
datasets/DIS5K_prepared   (output: canonical size + gt_edge + gt_brush)
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
4. **Writes `gt_edge/`** — the clean line, no RNG involved.
5. **Writes `gt_brush/`** — one synthetic brush stroke per mask, seed 0, so the
   validation set is reproducible.
6. **Writes `meta.json`** with per-split counts, rejection reasons and how many
   rasters were written.

**Cropping is not a preprocessing step.** The training crop is sampled on the fly
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
--force                     # regenerate existing gt_edge and gt_brush
--no-edge                   # skip gt_edge
--no-brush                  # skip gt_brush
```

### Verify before training

```bash
python tools/verify_alignment.py --dataset datasets/DIS5K_prepared
```

```
train   2998 images | shape 0 | non-binary 0 | oversized 0 | no-edge 0 | no-brush 0 | edge-on-boundary 1.000 | edge-covered-by-brush 0.970 | brush-on-edge 0.297
val      470 images | shape 0 | non-binary 0 | oversized 0 | no-edge 0 | no-brush 0 | edge-on-boundary 1.000 | edge-covered-by-brush 0.972 | brush-on-edge 0.310
test    1999 images | shape 0 | non-binary 0 | oversized 0 | no-edge 0 | no-brush 0 | edge-on-boundary 1.000 | edge-covered-by-brush 0.970 | brush-on-edge 0.310
```

| Column | Must be | Meaning |
|---|---|---|
| `shape` | 0 | image, mask and both rasters share dimensions |
| `non-binary` | 0 | masks are strictly {0, 255} |
| `oversized` | 0 | the split was preprocessed |
| `edge-on-boundary` | 1.000 | every `gt_edge` pixel is on the mask boundary |
| `edge-covered-by-brush` | ≈ 1.0 | every `gt_edge` pixel has a brush pixel within 8 px — the brush spans the whole edge |
| `brush-on-edge` | < 0.6 | the share of brush pixels that land on the edge — the stroke is rough, not a copy |

A contact sheet lands in `datasets/_qa/alignment/alignment_<split>.jpg`:
green = clean edge, orange = brush, blue = mask contour.

---

## Step 3 — Train and validate

`config.py` is the only config file: defaults, types and validation all live
there. Set the dataset by editing it:

```python
dataset: str = "DIS5K_prepared"
width: int = 20        # 16 = faster, 32 = more accurate
batch_size: int = 32
epochs: int = 60
```

or override for a single run without touching the file:

```bash
python -m training.train --set dataset=DIS5K_prepared --set epochs=10
```

Then:

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
because the product needs the line in the right place, not merely a plausible
mask.

Validation crops are always boundary-centred. Training crops use the four-way
mixture in `config.py` (`mix_boundary`, `mix_near`, `mix_empty`, `mix_dense`);
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

### The editor

The site opens straight into the editor with an empty canvas. There is no dataset
browser: you upload your own image, which is stored under `upload_dataset`
(`datasets/uploads/` by default) and opened for editing. **Open** lists what you
have already uploaded.

1. **Upload** — the stage asks for an image until one is loaded.
2. **Draw** — the magic brush (`M`) is the default tool. Drag *roughly* along
   the object edge. The stroke is the whole request: it does not matter how fast
   you drag or how long the stroke is, the server always works on one square
   around it.
3. **The line lands in the mask** — on release, the smallest square containing
   the whole stroke is cut out, run through the UNet once with the stroke itself,
   then clipped to the stroke you drew, thinned and re-widened into a **uniform
   clean line**. There is no accept step: the line is the result, so it goes into
   the mask immediately. A stroke you do not want is one `Ctrl+Z` away.
4. **Fill** (`F`) — flood the region bounded by the line: hairline gaps up to
   `fill_close` px are bridged, the outside is flooded, holes stay holes, and the
   boundary is refined with a guided filter before `expand` is applied.
5. **Save** — writes `<dataset>/<split>/gt/<name>.png` at the image's native size.

`Alt`+drag subtracts instead of adding. `Ctrl+Z` undoes, including a fill.

### One brush width, one thinner prompt

The brush you draw is a single width, and that is the only width on screen: the
cursor is one ring, the trail is one translucent stroke, and the slider reads one
number.

Behind it the model is conditioned on a thinner stroke, because that is what it
was trained on:

| | Width | Where it comes from |
|---|---|---|
| **Brush** (what you draw and see) | 32 px by default | the *Brush width* slider, 8–96 px |
| **Prompt** (the mask sent) | 12 px | `magic_prompt_ratio` (0.625) of the brush, clamped to `[magic_prompt_min, magic_prompt_max]` = [3, 12] |

The prompt is painted into the stroke mask and never drawn over the image, so
there is no inner core to line up with a crop any more. The difference between
the two widths is the tolerance the predicted line may move in — that is what
keeps the clean line attributable to the stroke: a line that drifts 8 px off the
centreline is still visibly *your* stroke, while a line 8 px outside a hairline
stroke looks like it came from nowhere.

The tolerance is enforced, not hoped for. `/api/segment` masks the edge with the
stroke (inset by half the output line width) before thinning it, so the returned
line cannot contain a pixel outside the drawn stroke. When the model wanders past
the stroke, the line is trimmed and `stats.clipped_px` reports it; the UI
suggests widening the brush when more than 30 % of the prediction was cut.

`magic_prompt_max` defaults to 12 because that is the widest brush the model
was trained on. Raising it makes the prompt thicker — closer to "the stroke I
drew is what the model sees" — but pushes the fourth channel out of distribution.

### One square, one forward pass

The stroke decides the crop. `/api/segment` takes the bounding box of the painted
prompt, grows it by half the brush on every side, and makes it square:

```
side = max(bbox_width, bbox_height) + brush_width      →  snapped up to a multiple of 32
```

| Stroke | Window | Run at |
|---|---|---|
| a 40 px dot | 64 px (floored at `crop_min`) | 64 |
| a 150 × 100 px arc | 192 px | 192 |
| an outline spanning 600 px | 640 px | 256 (downscaled 0.4×) |

The window is the *smallest* square that holds the whole stroke, so a short
stroke is not padded out to the canvas and a long one is not chopped into tiles.
That means exactly one UNet call per stroke, no seam where two crops disagreed,
and no merge to average.

Two sizes are reported because they differ for long strokes: `window.size` is the
square in canvas pixels, `crop_size` is what the network ran at. A window above
`crop_size` is **downscaled**, never upscaled — 256 px is the only scale the
model ever saw, and a short stroke blown up would turn its 12 px prompt into a
30 px blob. Downscaling shrinks the prompt too, which keeps it inside the 3–12 px
range it was trained on. The inspector shows `640 → 256 px` when that happens.

The canvas always works at the canonical short side (640), even when the stored
image is 4K, so the object scale at inference matches training.

**It starts without a checkpoint.** With no `model/bam_v1.pth`, the registry falls
back to a mock backend that predicts a band around your stroke, so the whole path
can be exercised. `/api/health` reports `model: mock` and the UI shows an amber
warning. Nothing is silently faked.

### API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | status, backend, device, data tier |
| `GET` | `/api/config` | crop bounds, line width, fill tolerance |
| `GET` | `/api/datasets` | datasets with per-split counts |
| `GET` | `/api/datasets/<ds>/images?split=train` | image names |
| `GET` | `/media/<ds>/<split>/<im\|gt\|gt_edge\|gt_brush>/<name>` | image bytes |
| `POST` | `/api/segment` | stroke → predicted line (+ `closed`) |
| `POST` | `/api/fill` | line in the mask → filled region |
| `POST` | `/api/mask/save` | save a mask |
| `POST` | `/api/image/upload` | upload and canonicalise |
| `POST` | `/api/magic_brush/predict_crops` | reference-client contract, returns an edge mask |
| `POST` | `/api/magic_brush/predict_single_crop` | single crop, for debugging |

`/api/segment` takes the stroke **mask**, not crop images or crop boxes: the
server reads its bounding box and cuts the window itself. That removes a class of
base64 alignment bugs, cuts the payload roughly 20×, and means the client cannot
ask for a window that misses part of the stroke it just drew.

```json
// POST /api/segment  ->
{
  "status": "success",
  "line_base64": "data:image/png;base64,...",   // the predicted clean line
  "mask_base64": null,                          // only in mode="object"
  "edge_base64": "data:image/png;base64,...",   // 1-px skeleton, for debugging
  "closed": true,                               // can this line be filled?
  "window": { "x0": 24, "y0": 34, "size": 192 },// the square, in canvas px
  "crop_size": 192,                             // what the network ran at
  "scale": 1.0,                                 // crop_size / window.size
  "band_width": 32,                             // the stroke the user drew
  "prompt_width": 20,                           // the thinner mask the model saw
  "stats": { "num_crops": 1, "ms_infer": 24.1, "ms_post": 21.7, "clipped_px": 0 }
}
```

---

## Configuration

One flat list, one setting per line. See `config.py` for all of them.

```yaml
dataset: demo                 # folder under datasets_dir
datasets_dir: ./datasets
short_side: 640               # downscale target
crop_size: 256                # training crop, and the largest inference window
crop_min: 64                  # smallest inference window
width: 20                     # UNet base channels: 16 / 20 / 24 / 32
depth: 4
batch_size: 32
epochs: 60
lr: 0.0003
mode: line                    # line = predict the edge, object = also fill
line_width: 3                 # predicted line thickness, canvas px
fill_close: 5                 # px of gap bridged before filling
magic_brush_width: 32         # visible brush, canvas px (slider default)
magic_prompt_ratio: 0.625     # prompt stroke / brush width
magic_prompt_min: 3           # prompt is never thinner than this
magic_prompt_max: 12          # ...nor wider
upload_dataset: uploads       # where the site stores uploaded images
train_device: auto
infer_device: cpu
backend: auto
```

Override without editing the file:

```bash
BAM_BATCH_SIZE=16 python run.py
python run.py --set port=8001 --set width=24
```

Resolution order: CLI > environment > defaults in `config.py`.

---

## How the model works

**Input** `4 × H × W`: RGB plus the brush channel.
**Output** `1 × H × W`: a soft edge band.

The target is a Gaussian-blurred `gt_edge` (σ = `target_sigma`), not a 1-px line —
a 1-px line is 1–3% positive pixels and a 1-px miss is a total loss. The band
keeps the peak exactly on the edge while giving a graded penalty nearby, which is
what makes the target trainable under class imbalance. Loss is
`0.5·BCE(pos_weight=6) + 0.5·Dice + 0.1·L1`, masked by a validity map so padded
crop regions contribute nothing.

The architecture is a UNet with depthwise-separable blocks, width 20, depth 4:
**0.34 M parameters, 1.58 GFLOPs** at 256². Measured on a 16-core CPU with
`channels_last` and 8 threads: **24 ms/crop** at batch 1.

**One window per stroke** means there is nothing to merge: the single prediction
is pasted into a full-canvas probability map at the window's offset, so a stroke
costs one forward pass instead of a handful. `PredictionMerger` is still the
paste mechanism (and still serves the reference endpoint), but its averaging and
brush-boost paths no longer run for the editor.

**Edge → line** is deterministic: threshold, thin to a 1-px skeleton
(Zhang-Suen), then dilate to `line_width`. Thinning first means the line is
uniform even where the brush pressure varied.

**Line → mask** is deterministic:

1. Close hairline gaps up to `fill_close` px, then dilate into a barrier so the
   flood cannot leak through a 1-px hole.
2. Label free space; the object is the enclosed component **adjacent to the
   line**. Other enclosed components are holes and are excluded, so donuts and
   windows survive.
3. Refine with a guided filter, then clear the holes again, then apply `expand`.
4. An open line cannot define a region, so the endpoint returns the line and
   reports `closed: false` rather than guessing a blob.

---

## Repository layout

```
BAM/
├── datasets/            # <name>/{train,val,test}/{im,gt,gt_edge,gt_brush} + meta.json
├── model/               # trained weights (bam_v1.pth / .onnx)
├── frontend/
│   ├── templates/       # base, index
│   └── static/{css,js}  # canvas.js, magic_brush.js, settings.js, ...
├── backend/
│   ├── app.py           # Flask factory
│   ├── routes/          # pages, datasets, masks, magic_brush, health
│   ├── inference/       # crop_window, predictor, merger, edge_to_mask, registry
│   └── services/        # storage, dataset_index, cache
├── training/
│   ├── prepare/         # download, preprocess, dedupe
│   ├── unet.py          # BAMUNet, self-contained
│   ├── paths.py         # split scanning, torch-free
│   ├── stroke_synth.py  # clean_edge + synthesize_brush: the data IP
│   ├── dataset.py, sampler.py, losses.py, metrics.py
│   └── train.py, evaluate.py, export_onnx.py
├── tools/               # make_demo_subset, verify_alignment, visualize_crops, benchmark_cpu
├── tests/               # 163 tests
├── config.py            # the only config: flat dataclass + defaults
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

- **Thin structures** (wires, cracks): the line is the product. Fill adds little,
  and the predicted line at `line_width` is usually the right answer.
- **Transparent or heavily occluded objects**: boundaries are genuinely ambiguous.
- **Objects under ~16 px** at the working resolution: below the model's useful scale.
- **Open lines** cannot be filled; the fill reports that instead of guessing.
- **One object per loop**: multiple loops in one pass are unioned.
- **A very thick brush over-fills**: the region covers the line itself, so a 20 px
  stroke adds ~10 px around the object. The predicted line is 3 px, so in practice
  the bias is sub-pixel.

---

## Testing

```bash
pytest -q
python -m pyflakes backend training tools tests config.py imgio.py run.py
```

163 tests cover config precedence, the stroke→window geometry (the square holds
the whole stroke, is centred on it, never upscales, and is capped at the trained
crop), edge/brush synthesis invariants (the clean edge sits on the boundary and
covers the whole perimeter; the brush is not the edge), joint-transform
alignment, crop-bucket coverage, model shapes and checkpoint round-trips, merge
arithmetic, the fill algorithm (hole preservation, open-line handling, gap
bridging), the API contract across splits, and path-traversal rejection.
