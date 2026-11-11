# BAM — Brush Anything Model

Stroke-guided edge segmentation for any object. You draw a rough line along the boundary; a small 4-channel UNet (RGB + stroke) snaps it to the true edge, and a deterministic post-process turns that edge into a filled mask.

Runs on CPU. No GPU required at inference.

```
stroke → 256×256 crops → UNet (0.34M params) → merge → edge → fill → mask
```

---

## What it is

BAM is a **local edge refiner**, not a global segmenter. The model never sees the whole image. It answers one question, 8-64 times per stroke:

> Given this 256×256 patch and this rough stroke inside it, where is the real boundary?

That framing is what makes a sub-million-parameter model viable at interactive speed, and it is why the tool is class-agnostic: it has no idea what an object *is*, only where its edge is.

The web app is a fork of the [masker](https://github.com/howardtse2005/masker) workflow (Flask + vanilla JS canvas + the `PredictionMerger` magic pen), upgraded from cracks to arbitrary objects.

---

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
# CPU-only PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Optional: a synthetic dataset so the app has something to show
python tools/make_demo_subset.py --count 24
python -m training.prepare.normalize --name demo --source datasets/_raw/demo --tier A
python -m training.prepare.split

python run.py
```

Open <http://127.0.0.1:8000>.

**The app starts without a checkpoint.** The model registry falls back to a mock backend that predicts a band around your stroke, so the full request path, merge, fill and UI can be exercised before training. `/api/health` reports `model: mock` and the UI shows a warning. Nothing is silently faked.

---

## Training

### 1. Prepare data

BAM expects a flat folder of image/mask pairs:

```
<source>/images/0001.jpg
<source>/masks/0001.png
```

Then normalise into the canonical layout:

```bash
python -m training.prepare.normalize --name coco --source datasets/_raw/coco_train2017 --tier A
python -m training.prepare.split --ratios 0.8 0.1 0.1
python tools/verify_alignment.py
```

`verify_alignment.py` is not optional. It asserts that RGB, GT and the synthetic stroke share one shape and origin, and writes a contact sheet to `datasets/_qa/alignment/`. Run it before every training run.

COCO can be downloaded and converted automatically:

```bash
python -m training.prepare.download --coco --coco-split val2017
```

### 2. Train

```bash
python -m training.train --dry-run          # one synthetic batch, no data needed
python -m training.train                    # full run from config.yaml
python -m training.train --set train.epochs=20 --set train.batch_size=16
```

Training uses AdamW, a cosine schedule with warmup, AMP on CUDA, EMA weights, and early stopping on validation boundary F@2px. The best checkpoint is copied to `model/bam_v1.pth`.

On a 24 GB GPU, 60 epochs over 24,000 crops each takes roughly 1.5-3 hours. On CPU it is 10-25 hours: viable overnight, not for iteration.

### 3. Evaluate

```bash
python -m training.evaluate --split test_a --checkpoint model/bam_v1.pth
python -m training.evaluate --split test_b --checkpoint runs/.../best.pth
```

### 4. Export for CPU

```bash
python -m training.export_onnx --checkpoint model/bam_v1.pth --int8
```

The backend prefers `onnx_int8` → `onnx_fp32` → `torch` → `mock`, in that order, and verifies parity before accepting a backend.

---

## How it works

### Synthetic strokes

The model needs to learn "rough human stroke → precise boundary". Training data for that does not exist, so it is generated from binary masks:

1. Extract ordered contours with `findContours`.
2. Simplify (RDP, epsilon 2-10 px) — users cut corners.
3. Smooth (Hann window 5-21).
4. Deform along the local normal: global offset, two sinusoids, fractal noise, clipped to 12 px.
5. Choose 1-3 arcs covering 45-100% of the perimeter; sometimes only a short scribble.
6. Rasterise with a variable-radius anti-aliased brush, occasional dry-brush gaps.
7. 15% of samples also get a distractor stroke along a strong image gradient that is *not* the boundary.

Strokes are regenerated on the fly for every training sample (`data.stroke_source: online`), so augmentation is effectively unbounded. `gt_edge/` on disk is the reproducible seed-0 variant used for QA.

### Edge → mask

The network predicts an edge; a product needs a mask. `backend/inference/edge_to_mask.py` does this deterministically:

1. Threshold and thin the edge (Zhang-Suen, implemented locally — no scikit-image).
2. Dilate into a 3 px barrier so the fill cannot leak through 1 px gaps.
3. Label the free space; the object is the enclosed component **adjacent to the stroke ring**. Holes are separate components and are therefore preserved.
4. Refine with a guided filter so the mask snaps to image edges.
5. Fall back to GrabCut, then to the raw loop, if the barrier leaks.

An open stroke cannot define a region, so the endpoint returns the predicted edge and flags `mode_used: "open_loop"` rather than guessing a blob.

---

## Configuration

`config.py` defines the schema; `config.yaml` holds defaults. Resolution order:

```
CLI --set  >  environment  >  config.yaml  >  dataclass defaults
```

```bash
python run.py --set infer.device=cuda
BAM_INFER__DEVICE=cuda python run.py        # double underscore = section separator
```

Key settings:

| Key | Default | Meaning |
|---|---|---|
| `infer.device` | `cpu` | inference device (`cpu` / `cuda`) |
| `train.device` | `auto` | training device |
| `infer.backend` | `auto` | `auto` / `onnx_int8` / `onnx_fp32` / `torch` / `mock` |
| `data.canonical_short_side` | `640` | working resolution |
| `data.crop_size` | `256` | must be a multiple of 32 |
| `infer.stride` | `128` | crop spacing, 50% overlap |
| `infer.max_crops` | `64` | per-request cap |
| `model.width` | `20` | base channel width |
| `infer.allow_mock` | `true` | allow the placeholder backend |

The resolved config is embedded in every checkpoint, so a model is always reproducible from its weights.

---

## Repository layout

```
BAM/
├── datasets/            # <name>/{rgb,gt,gt_edge} + manifest.json
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
│   ├── prepare/         # download, normalize, split, dedupe
│   ├── unet.py          # BAMUNet, self-contained
│   ├── stroke_synth.py  # the data IP
│   ├── dataset.py, sampler.py, losses.py, metrics.py
│   └── train.py, evaluate.py, export_onnx.py
├── tools/               # verify_alignment, visualize_crops, benchmark_cpu
├── tests/               # 96 tests
├── config.py            # dataclass config
├── run.py
└── README.md
```

`model/` holds weights only. The architecture lives in `training/unet.py` and imports nothing from `config`, so it can be loaded standalone. The backend imports it lazily, inside `TorchBackend`, so the web app still starts when PyTorch is not installed.

---

## Licensing

| Dataset | Status |
|---|---|
| **Open Images V7** | images CC BY 2.0, masks CC BY 4.0 — commercial use with attribution |
| **COCO 2017** | annotations CC BY 4.0; images under Flickr terms, verify per image |
| **DIS5K** | no explicit licence — all rights reserved by default |
| **DUTS** | *"All rights reserved by the original authors"* |
| **ECSSD / HKU-IS / DUT-OMRON** | research datasets, no commercial grant |
| **SA-1B** | SA-1B Dataset Research License |
| **Cityscapes / Mapillary / BDD100K / Objects365 / xView** | non-commercial or research |

BAM therefore uses two tiers:

- **Tier A** (`data.tier: "A"`): Open Images + COCO. This is the shippable checkpoint.
- **Tier B** (`data.tier: "B"`): add DIS5K/DUTS/ECSSD for benchmarks and ablations.

Every checkpoint records its tier, and the backend **refuses to load a tier-B checkpoint unless the config explicitly says `data.tier: B`**. That is a guard rail, not a formality.

Attribution is a legal obligation. If you redistribute the app, keep the credits in this section.

---

## Testing

```bash
pytest -q
```

96 tests cover config precedence, stroke-synthesis invariants, joint-transform alignment, crop-bucket coverage, model shapes and checkpoint round-trips, merge arithmetic, the fill algorithm (including hole preservation and open-loop handling), the API contract, and path-traversal rejection.

```bash
python tools/verify_alignment.py      # dataset contract
python tools/visualize_crops.py       # what the model actually sees
python tools/benchmark_cpu.py --random
```

---

## Citation

If you use the magic-pen workflow, cite the original masker project. If you use the datasets, cite them and check their licences first.
