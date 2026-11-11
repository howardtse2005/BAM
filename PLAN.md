# BAM — Brush Anything Model

**Technical Plan v1.0**

> Status: proposal for approval. No implementation code in this stage.
> Scope: data pipeline, model, training, inference (magic pen), backend, frontend, evaluation, milestones.
> Reference implementation studied: [`howardtse2005/masker`](https://github.com/howardtse2005/masker) (Flask + vanilla JS canvas + `PredictionMerger` magic pen).

---

## Table of contents

1. [What BAM is](#1-what-bam-is)
2. [System overview](#2-system-overview)
3. [Data](#3-data)
4. [Model](#4-model)
5. [Inference: the magic pen](#5-inference-the-magic-pen)
6. [Backend](#6-backend)
7. [Frontend](#7-frontend)
8. [Evaluation](#8-evaluation)
9. [Training operations](#9-training-operations)
10. [Milestones](#10-milestones)
11. [Risks](#11-risks)
12. [Decisions log](#12-decisions-log)
13. [Open questions](#13-open-questions)
- [Appendix A — Config reference](#appendix-a--config-reference)
- [Appendix B — API schemas](#appendix-b--api-schemas)
- [Appendix C — Findings in masker to fix](#appendix-c--findings-in-masker-to-fix)
- [Appendix D — File naming and manifest](#appendix-d--file-naming-and-manifest)
- [Appendix E — Glossary](#appendix-e--glossary)

---

## 1. What BAM is

### 1.1 One sentence

BAM is a **class-agnostic, local edge refiner**: the user draws a rough stroke along the boundary of any object, and a small 4-channel UNet (RGB + stroke) snaps that stroke onto the true boundary; a deterministic post-process converts the predicted edge into a filled mask.

### 1.2 The interaction contract

```
user draws a rough loop  ->  crops along the stroke  ->  UNet predicts edge per crop
                                                              |
                          filled mask  <-  edge -> region  <-  merged edge map
```

The model never has to understand the whole image. It only has to answer one local question, 256 times per stroke:

> "Given this 256x256 patch of image and this rough stroke inside it, where is the real object boundary?"

This is the key design property. It makes a sub-million-parameter model viable on CPU, and it makes the model scale-agnostic (it is fully convolutional).

### 1.3 Why the problem is well-posed

- The stroke is a **spatial prior**: it removes the global search problem. The model only needs local precision.
- The stroke is **not the answer**: it is jittered, simplified, thickened and often offset by 2-8 px. Identity mapping (`output = input`) is explicitly punished during training.
- The boundary is **locally determined by appearance**: intensity/color/texture discontinuity. This is learnable from a small receptive field.
- The remaining ambiguity (which side is foreground) is resolved **geometrically** at the global level from the closed stroke loop, not by the network. This is why the model can stay single-head.

### 1.4 Non-goals (v1)

- No text/box/click prompts. Stroke only.
- No semantic labels. Class-agnostic binary masks.
- No video. No 3D. No batch server inference.
- No foundation-model backbone. No SAM distillation in v1.
- No guarantee on transparent, heavily occluded, or sub-16 px objects (documented limits, see §11).

---

## 2. System overview

### 2.1 Two pipelines

**Offline (training)** — runs once, on GPU:

```
raw datasets -> normalize -> canonical rgb/gt -> synthetic gt_edge -> manifest
                                                                        |
                                              crop sampler -> UNet -> checkpoint
```

**Online (serving)** — runs per stroke, on CPU:

```
browser stroke -> crops + stroke raster -> batched UNet -> merge -> edge->mask -> PNG
```

### 2.2 End-to-end data flow

```
┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐
│  Dataset GT  │──▶│ Boundary extract  │──▶│ Stroke synthesis │──▶ gt_edge/
│  (binary)    │   │ skeleton + contour│   │ jitter/wobble/   │
└──────────────┘   └───────────────────┘   │ radius/coverage  │
                                            └──────────────────┘
┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐
│  RGB image   │──▶│ Canonical resize  │──▶│ 256x256 crop     │──▶ x: 4x256x256
│              │   │ short side 640    │   │ joint sampler    │    y: 1x256x256
└──────────────┘   └───────────────────┘   └──────────────────┘
                                                     │
                                          ┌──────────▼──────────┐
                                          │  BAMUNet (0.33M)    │
                                          │  4ch -> 1ch logits  │
                                          └──────────┬──────────┘
                                                     │
   browser ──crops──▶ /api/segment ──▶ batch forward ──▶ merge ──▶ edge→mask ──▶ PNG
```

### 2.3 Repository layout

Keeps the requested top level (`datasets/`, `model/`, `frontend/`, `backend/`, `config.py`, `README.md`). `model/` holds **weights only**. The architecture lives in `training/unet.py`: it is model code, not a runtime entry point, so it belongs with the rest of the model code rather than at the repository root.

> **Revision (implementation).** Datasets are organised by split rather than by
> kind: `<name>/{train,val,test}/{im,gt,gt_edge}/`. The `rgb/` + manifest design
> below was replaced because the split is a property of the data the user already
> has, and a manifest that restates the folder tree is one more thing to keep in
> sync. `gt_edge/` is still generated by preprocessing, just per split.

```
BAM/
├── datasets/
│   ├── _raw/                      # untouched sources, gitignored
│   ├── <name>/
│   │   ├── train/{im,gt,gt_edge}/
│   │   ├── val/{im,gt,gt_edge}/
│   │   ├── test/{im,gt,gt_edge}/
│   │   └── meta.json              # written by preprocessing
├── model/
│   ├── bam_v1.pth
│   ├── bam_v1.onnx
│   └── bam_v1_int8.onnx
├── frontend/
│   ├── templates/{base.html, select_dataset.html, index.html}
│   └── static/
│       ├── css/{tokens.css, main.css, components.css}
│       ├── js/{main.js, canvas.js, magic_pen.js, ui.js, settings.js,
│       │        file_system.js, toast.js, shortcuts.js}
│       └── img/
├── backend/
│   ├── app.py                     # Flask factory + blueprint registration
│   ├── routes/{pages.py, datasets.py, masks.py, magic_pen.py, health.py}
│   ├── inference/{predictor.py, merger.py, edge_to_mask.py, registry.py}
│   ├── services/{storage.py, dataset_index.py, cache.py}
│   └── schemas.py
├── training/
│   ├── prepare/{download.py, preprocess.py, dedupe.py}
│   ├── unet.py                    # BAMUNet + build_model()
│   ├── paths.py                   # split scanning, torch-free
│   ├── stroke_synth.py            # the core data IP
│   ├── dataset.py
│   ├── sampler.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
├── tools/
│   ├── verify_alignment.py
│   ├── visualize_crops.py
│   ├── benchmark_cpu.py
│   └── make_demo_subset.py
├── tests/
├── config.py                      # dataclasses, YAML/env/CLI merge
├── config.yaml                    # user-editable defaults
├── run.py                         # entry point
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

Flask is configured with `template_folder="../frontend/templates"` and `static_folder="../frontend/static"` so the requested split is preserved.

---

## 3. Data

### 3.1 Licensing first — an important correction

The brief lists DIS5K and DUTS as commercially safe. They are not, and this affects what the shipped weights may be trained on.

| Dataset | Image license | Mask license | Commercial use | Verdict |
|---|---|---|---|---|
| **Open Images V7** (instance masks) | CC BY 2.0, per-image (Google asks you to verify) | CC BY 4.0 (Google LLC) | **Yes, with attribution** | Primary corpus |
| **COCO 2017** (instance masks) | Flickr terms, mixed CC per image | CC BY 4.0 (COCO Consortium) | Annotations yes; images must be checked per image | Primary corpus |
| **DIS5K** | Flickr images, no explicit license | **No explicit license** | **No** — no grant means all rights reserved | Research / eval only |
| **DUTS** | ImageNet + SUN images | Site states: *"All rights reserved by the original authors"* | **No** | Research / eval only |
| **ECSSD, HKU-IS, DUT-OMRON** | Research datasets | Research datasets | No explicit commercial grant | Eval only |
| **SA-1B** | SA-1B **Research** License | Same | No | Excluded |
| **ADE20K** | SUN/Places derived, separate terms | ADE20K terms | Must be verified case by case | Optional, verify |
| **Cityscapes, Mapillary, BDD100K, Objects365, xView** | Non-commercial / research | Non-commercial / research | No | Excluded |

**Consequence — two tracks, one codebase:**

- **Track A (shippable).** Train only on Open Images V7 + COCO. This is the checkpoint that goes in `model/` and ships with the app.
- **Track B (research).** Add DIS5K/DUTS/ECSSD/HKU-IS for benchmark numbers and ablations. Never ship these weights, never publish them without a license review.

If BAM stays a personal/research project, Track B is fine. The plan supports both via `data.tier: "A" | "B"` and a `data_license` field baked into every checkpoint, so a shipped model can be audited.

Attribution requirements (CC BY) must appear in `README.md` and in an in-app "About / Licenses" panel. This is a legal obligation, not a courtesy.

**If more clean volume is needed:** CC0/CC BY images from Openverse, Wikimedia Commons, or Flickr CC BY, pseudo-labelled and then manually verified on a sample. Note that SAM checkpoints carry their own license terms — do not assume Apache 2.0 covers the weights.

### 3.2 Recommended corpus

| Tier | Dataset | Images | Instances | Role |
|---|---|---|---|---|
| A | Open Images V7 instance masks | ~60,000 | ~180,000 | Category and scene diversity |
| A | COCO 2017 train | ~40,000 | ~120,000 | Multi-object, clutter, small objects |
| A | COCO 2017 val | 5,000 | — | Held-out test (never trained on) |
| B | DIS5K (VD + TE) | 470 + 2,000 | — | High-precision benchmark |
| B | DUTS-TE | 5,019 | — | Saliency benchmark |
| B | ECSSD / HKU-IS / DUT-OMRON | 1,000 / 4,447 / 5,168 | — | Legacy comparability |
| C | In-house human strokes | 200 | 200 | Real-vs-synthetic calibration (§8.6) |

Open Images and COCO overlap in content. Deduplicate across datasets before splitting (§3.4).

### 3.3 Canonical resolution and the alignment contract

**Rejected option: downscale everything to the lowest-resolution dataset.** DUTS is ~400 px short side. Matching it would discard 4-10x detail from DIS5K and Open Images, and — more importantly — it would change *what the model is asked to learn*. At 400 px short side, a 256 crop contains most of the object, so the network learns global object shape. At inference, a crop contains a local boundary segment. That mismatch is fatal.

**Chosen option: normalize object scale, not file size.**

| Parameter | Value | Rationale |
|---|---|---|
| Canonical short side | **640 px** | A 256 crop covers ~40% of the short side: a local boundary segment, matching inference. |
| Upscaling | **Never above 1.0x** during normalization | Avoid inventing detail. |
| Images already smaller than 640 | Kept native; compensated by crop-time zoom augmentation | DUTS stays ~400 px and is still usable. |
| Training crop | 256 x 256 | Fixed. Matches model input. |
| Inference working size | 640 px short side (configurable 1024 for high-res mode) | Train/test scale match by construction (§5.2). |

**The alignment contract.** For every sample, three arrays must be pixel-identical in shape and origin:

```
rgb[i].shape == gt[i].shape == gt_edge[i].shape == (H_i, W_i)
```

Rules:
1. Resize once, from the original, with `INTER_AREA` for RGB and `INTER_NEAREST` for masks.
2. Never resize `gt` or `gt_edge` independently of `rgb`.
3. Never re-crop one array without applying the identical integer offsets to the other two.
4. All geometry is recorded in `manifest.json` (original size, scale, crop box) so any sample can be reproduced.

`tools/verify_alignment.py` enforces this: it walks every dataset, asserts shape equality, asserts `boundary(gt) ⊆ dilate(gt_edge, r_max + 3)`, and writes a 200-sample visual contact sheet to `datasets/_qa/alignment/`. This runs in CI on a 200-image subset and must pass before any training run.

### 3.4 Preprocessing pipeline

`training/prepare/` runs in this order, idempotent, resumable, logged to `datasets/manifest.json`:

1. **Download** (`download.py`) — fetch archives, verify checksums, unpack to `_raw/`.
2. **Extract instances** — COCO/Open Images annotations are per-instance; render each instance to a binary mask, crop to its bbox with 32 px context, drop instances with area < 400 px or > 95% of the frame.
3. **Normalize** (`normalize.py`) — resize to canonical short side 640 (downscale only), write `rgb/` as JPEG q95 and `gt/` as PNG (0/255).
4. **Dedupe** (`dedupe.py`) — perceptual hash (pHash, 64-bit). Hamming distance <= 6 means duplicate; duplicates are grouped and assigned to the same split. Also catches COCO/Open Images cross-contamination.
5. **Split** (`split.py`) — 80/10/10 **by source image**, stratified by dataset and object-size bucket. Never split by crop: crops from one image leak trivially.
6. **Synthesize strokes** (`stroke_synth.py`) — write `gt_edge/` with seed 0 per image, for inspection and for the manifest (§3.5).
7. **QA** — contact sheets, histogram of stroke radius, boundary coverage, and a rejection list.

Rejection criteria (logged, not silently dropped):
- GT has < 200 foreground pixels or > 98% foreground.
- GT boundary is empty after skeletonization.
- RGB and GT aspect ratios differ by > 1%.
- Mean gradient magnitude along the GT boundary < 8/255 (blurry or wrong mask — see §3.9).

### 3.5 Synthetic stroke generator

This is the most important piece of the data pipeline: the model's real-world accuracy is bounded by how faithfully `gt_edge` resembles a human stroke. It runs at canonical scale.

**Step 1 — boundary extraction.**
- `cv2.findContours(gt, RETR_CCOMP, CHAIN_APPROX_NONE)` to keep inner contours (holes).
- Rasterize contours to a 1 px band, then `skimage.morphology.skeletonize` to a true 1 px centerline.
- Keep contours with arc length >= 32 px.

**Step 2 — path simplification (user imperfection).**
- Resample the centerline at 6 px spacing.
- Ramer-Douglas-Peucker with `epsilon ~ U(2, 10)` px: users cut corners.
- Moving-average smoothing with window `U(5, 21)`.
- Result: a polyline that is recognizably the boundary but not a tracing.

**Step 3 — deformation (jitter and wobble).**
Displace each point along the local normal by:

```
d(t) = d0 + a1*sin(2*pi*t/L1 + p1) + a2*sin(2*pi*t/L2 + p2) + perlin(t)
```

with `d0 ~ N(0, 3)` px global offset, `a1 ~ U(1.0, 4.0)`, `L1 ~ U(60, 180)` px, `a2 ~ U(0.5, 2.0)`, `L2 ~ U(20, 60)` px, and Perlin amplitude `U(1, 4)` px. Total displacement clipped to 12 px.

**Step 4 — coverage (user impatience).**
- Draw 1-3 disjoint arcs covering `U(0.45, 1.00)` of the perimeter.
- 15% of samples: two separate strokes, as if the user drew in two passes.
- 10% of samples: a short 60-120 px scribble only, to teach partial evidence.
- 5% of samples: the stroke drifts to the wrong side of the boundary by 4-10 px.

**Step 5 — brush rasterization.**
- Radius `r ~ U(1.5, 6.0)` px, modulated along the path by `1 + 0.25*sin(...)` to imitate pressure.
- Anti-aliased coverage: `alpha = clip((r - dist_to_path) / 1.5, 0, 1)`.
- 15% chance of a "dry brush" gap: drop 2-5 short segments.
- 15% chance of a distractor stroke along a strong image gradient that is **not** the object boundary.
- 5% chance of a stroke in the object interior.

**Step 6 — write.** `gt_edge/<stem>.png`, grayscale uint8, 0-255 alpha.

**Why the on-disk file is not the only stroke.** One deterministic stroke per image limits augmentation. Training therefore regenerates strokes **on the fly** from `gt` with a random seed (`data.stroke_source: "online"`, default), while `gt_edge/` remains the reproducible reference used for validation, visual QA, and the demo. `data.stroke_source: "disk"` reproduces the exact on-disk variant for debugging.

**Invariants** (unit-tested in `tests/test_stroke_synth.py`):
- Stroke pixels lie within `r_max + 12` px of the GT boundary, except for deliberate distractor/interior samples, which are flagged.
- Median brush radius in `[1.5, 6.0]`.
- Coverage distribution matches the configured mixture within tolerance over 1,000 samples.
- No stroke pixel outside the image.

### 3.6 Crop sampler

Sampling only boundary-centered crops teaches a degenerate shortcut: "draw a line through the center". The mixture below prevents that.

| Bucket | Share | Definition | Target |
|---|---|---|---|
| Boundary-centered | 50% | Center within 24 px of a boundary pixel | Boundary near center |
| Near-boundary | 20% | Center 32-96 px from the nearest boundary | Boundary off-center but visible |
| Empty | 15% | No boundary within 32 px of center | All zeros |
| Dense / clutter | 15% | >= 3 boundary pixels within 8 px, or >= 2 instances | Multiple boundaries |

Per batch, the mixture is enforced exactly (not sampled i.i.d.) so gradient variance stays stable.

Crops may extend past the image edge. Padding rules:
- RGB: `cv2.BORDER_REPLICATE` (never black — black invents a strong fake edge, a real bug in masker).
- Stroke channel: zeros.
- Target: zeros.
- Loss: multiplied by a validity mask so padded pixels do not contribute.

### 3.7 Augmentation

All geometric transforms are applied **jointly** to RGB, GT, stroke, and validity mask through a single affine matrix. `tests/test_joint_transform.py` asserts that a transformed boundary still coincides with the transformed GT boundary within 1 px.

| Transform | Range | Notes |
|---|---|---|
| Random scale | 0.75-1.35 | Applied before cropping; compensates for small source images |
| Horizontal / vertical flip | p=0.5 each | Objects have no canonical orientation |
| Rotation | 90 deg multiples, p=0.5 | Cheap and exact |
| Small rotation | +/- 12 deg, p=0.3 | Requires border replication |
| Brightness / contrast | +/- 0.2 | |
| Saturation / hue | 0.8-1.2 / +/- 0.05 | |
| Gaussian blur | sigma 0-1.2, p=0.2 | |
| Gaussian noise | sigma 0-6/255, p=0.2 | |
| JPEG artifacts | q 60-95, p=0.2 | Matches web images |
| Stroke regeneration | p=0.5 | New seed, new wobble |
| Stroke shift | +/- 6 px, p=0.2 | Hard-case robustness |
| Stroke dropout | p=0.08 | Forces use of RGB |
| Distractor stroke | p=0.1 | Forces discrimination |

### 3.8 Manifest and reproducibility

`datasets/manifest.json` is the single source of truth. Per sample:

```json
{
  "id": "openimages/000012345",
  "dataset": "openimages",
  "tier": "A",
  "rgb": "openimages/rgb/000012345.jpg",
  "gt": "openimages/gt/000012345.png",
  "gt_edge": "openimages/gt_edge/000012345.png",
  "original_size": [1024, 768],
  "canonical_size": [853, 640],
  "scale": 0.833,
  "split": "train",
  "instance_area_ratio": 0.31,
  "boundary_sharpness": 0.42,
  "quality_weight": 0.9,
  "license": "CC-BY-4.0-annotations/CC-BY-2.0-image",
  "stroke_seed": 0
}
```

The manifest hash is written into every checkpoint. A checkpoint without a manifest hash is not deployable.

### 3.9 Label quality

Masks differ a lot in quality. Training on all of them with equal weight teaches blurry boundaries.

- **Per-dataset weight:** Open Images 0.85, COCO 0.9, DIS5K 1.0, DUTS 0.6.
- **Per-image sharpness:** mean Sobel magnitude along the GT boundary, normalized. Below a threshold, downweight to 0.5 or reject.
- **Target bandwidth:** sigma 1.0 px for sharp masks, 1.5 px for coarse masks, so a coarse mask does not demand pixel-exact output.

This is the difference between a model that learns "object boundaries" and one that learns "the average of several annotators' guesses".

---

## 4. Model

### 4.1 Architecture

`training/unet.py` exports `BAMUNet` and `build_model(cfg)`. It must **not** import `config.py` at module scope — the masker's `model/unet.py` does (`from config import Config as cfg`), and `config.py` is missing from that repo, so the model cannot be imported standalone. BAM's model takes everything through constructor arguments.

```
input  4 x 256 x 256
  │
  ├─ enc1  DSBlock(4   -> 20)   256²   ────────────────┐ skip
  ├─ pool ─ enc2  DSBlock(20  -> 40)   128²   ────────┐│
  ├─ pool ─ enc3  DSBlock(40  -> 80)    64²   ──────┐ ││
  ├─ pool ─ enc4  DSBlock(80  -> 160)   32²   ────┐ │ ││
  ├─ pool ─ enc5  DSBlock(160 -> 256)   16²       │ │ ││
  │                                                │ │ ││
  ├─ up1: bilinear + 1x1(256->160), concat(320), DSBlock(320->160)  32²
  ├─ up2: bilinear + 1x1(160->80),  concat(160), DSBlock(160->80)   64²
  ├─ up3: bilinear + 1x1(80->40),   concat(80),  DSBlock(80->40)   128²
  ├─ up4: bilinear + 1x1(40->20),   concat(40),  DSBlock(40->20)   256²
  └─ head: 1x1(20 -> 1)  -> logits 256²
```

`DSBlock` = `3x3 depthwise -> BN -> ReLU -> 1x1 pointwise -> BN -> ReLU`, twice. Depthwise separable convolutions cut spatial cost by ~6-8x versus vanilla double-conv at the same width, which is what makes CPU inference viable. `block: "vanilla"` is available for ablation.

Design notes:
- **Bilinear upsampling + 1x1 reduce** instead of `ConvTranspose2d`: fewer parameters, no checkerboard artifacts.
- **MaxPool downsampling**: cheaper than strided conv, and the task is local.
- **BatchNorm** (not InstanceNorm): inference is batched and BN folds into the preceding conv at export, giving a free CPU speedup.
- **No pretrained backbone.** Training from scratch on ~100K images is enough for a local, low-level task, and it avoids a license dependency on ImageNet weights.

### 4.2 Sizing for CPU

| `width` | Params | GFLOPs @256² | Est. CPU ms/crop (8 cores, fp32) | Expected BF@2px |
|---|---|---|---|---|
| 16 | 0.22 M | 0.7 | ~22 | -0.5 vs baseline |
| **20 (default)** | **0.33 M** | **1.1** | **~32** | baseline |
| 24 | 0.48 M | 1.6 | ~45 | +0.4 |
| 32 | 0.85 M | 2.8 | ~80 | +0.8 |

These are estimates; `tools/benchmark_cpu.py` produces the real table in M2, and the width is frozen after the first full training run. The upgrade path if accuracy plateaus is `width: 32`, which still meets a ~1.9 s worst-case stroke budget.

FP32 checkpoint ~1.3 MB; INT8 ~0.35 MB. Both are small enough to ship inside the repo.

### 4.3 Output parameterization

Predicting a 1 px line is brittle: 1-3% positive pixels, and a 1 px miss is a total loss. Instead the target is a **soft boundary band**:

```
T = clip(gaussian_blur(boundary_1px, sigma=1.0), 0, 1)
```

This keeps the peak exactly on the true boundary but gives a graded penalty nearby. At inference, `T > 0.5` recovers a 2-3 px band, which is then skeletonized to 1 px.

Ablation item: **stroke encoding**. Default is a binary anti-aliased raster (spec-compliant). Alternative is a clipped signed distance field of the stroke mapped to [0,1], which gives the network an explicit "how far and which way" signal. If SDF beats binary by > 1.5 BF@2px in M3, it becomes the default and this section is updated.

### 4.4 Loss

```
L = 0.5 * BCEWithLogits(pos_weight=6) * w_quality * w_valid
  + 0.5 * (1 - SoftDice)
  + 0.1 * L1(sigmoid(logits), T)
```

- `pos_weight=6` counteracts the ~4-6% positive rate.
- `w_quality` is the per-image quality weight (§3.9).
- `w_valid` zeroes padded pixels.
- Dice is computed on the soft target, which is well-defined for graded labels.
- The L1 term sharpens the peak; without it the model tends to predict a wide, timid band.

Optional in M3: boundary IoU loss, focal loss with `gamma=2, alpha=0.75`. Keep v1 simple; add only if validation shows under-confidence.

### 4.5 Training recipe

| Item | Value |
|---|---|
| Crop | 256 x 256 |
| Batch | 32 |
| Crops per epoch | 24,000 (750 steps) |
| Epochs | 60 |
| Optimizer | AdamW, lr 3e-4, weight decay 1e-4 |
| Schedule | Cosine, 3-epoch linear warmup |
| Precision | bf16 autocast (fp16 fallback) |
| Gradient clip | 1.0 |
| EMA | 0.999, evaluated weights |
| Validation | 2,000 fixed crops, every epoch |
| Model selection | Boundary F@2px on validation |
| Early stop | patience 15 |
| Dataloader workers | 8 |

Two-stage sanity check before the full run:
1. **Overfit 32 images** to near-zero loss. If this fails, the bug is in the data, not the model.
2. **Train on 5,000 images for 10 epochs**, evaluate, and confirm the model beats a "copy the stroke" baseline. If it does not, the stroke augmentation is too weak or the target is misaligned.

### 4.6 Export and CPU optimization

`training/export_onnx.py`:
1. Fold BatchNorm into conv weights.
2. Export ONNX opset 17 with dynamic H/W axes (fully convolutional).
3. Verify parity: max abs diff < 1e-4 on 20 random inputs.
4. Dynamic INT8 quantization (`quantize_dynamic`, per-channel) and re-verify: max abs diff < 0.02 on logits, BF@2px drop < 0.5.

Runtime selection order in the backend: `onnx_int8` -> `onnx_fp32` -> `torch_fp32`. Configurable, with a startup self-test that compares each available backend against a stored golden output and refuses to load a backend that fails.

CPU settings: `torch.set_num_threads(min(8, physical_cores))`, `channels_last` memory format, `torch.inference_mode()`. Avoid oversubscription: one process, N threads, a single model instance, and a lock around forward.

---

## 5. Inference: the magic pen

### 5.1 End-to-end sequence

```
mousedown   -> start stroke, clear previous crop list
mousemove   -> interpolate points at 1 px spacing; accumulate arc length
            -> every 128 px of arc length, emit a crop request descriptor
mouseup     -> send one batched request (crops + full stroke raster + params)
server      -> batch forward -> merge -> edge->mask -> encode PNG
browser     -> composite mask, push undo state, show stats
```

Crops are emitted on **arc length**, not on mouse events, so a fast flick and a slow drag produce the same uniform coverage. This is a real improvement over masker, where crop spacing is tied to `line_len` and the counter logic is fragile.

### 5.2 Crop contract

| Parameter | Value | Notes |
|---|---|---|
| Crop size | 256 x 256 | Fixed; matches training |
| Stride | 128 px (50% overlap) | Uniform coverage, smooth blending |
| Max crops per request | 64 | Hard limit; excess is decimated |
| Coordinate space | **canonical** (short side 640) | Train/test scale match by construction |
| Padding at image edge | replicate RGB, zero stroke/target | No fake black edges |
| Encoding | WebP q90 (JPEG q92 fallback) | ~6-10x smaller than PNG |

**Why canonical coordinates.** The frontend loads a working image resized so its short side is 640 (downscale only). All strokes, crops, and returned masks live in that space. This removes every rounding and scale-mismatch class of bug, bounds CPU cost regardless of the uploaded file size, and guarantees the model sees the same object scale it was trained on. At export, the mask is upsampled to native resolution with a guided filter (radius 8, eps 1e-3) using the native image as guide, so edges stay crisp.

The stroke channel is **not** burned into the crop by the browser. The browser sends one full-canvas stroke raster plus crop boxes; the server crops the stroke with identical integer offsets. This guarantees the 4th channel is pixel-aligned with RGB, keeps the payload small, and gives the server the global stroke geometry it needs for `edge -> mask`.

### 5.3 Merge (preserved from masker)

The masker `PredictionMerger` semantics are kept: accumulate, average, boost by brush count, threshold, denoise, morphology.

For each crop `k` with predicted probability map `P_k` and canvas offset `o_k`:

```
A(p) += P_k(p - o_k) * v_k(p - o_k)
N(p) += v_k(p - o_k)
B(p) += v_k(p - o_k)

E(p)  = A(p) / max(N(p), 1)
E'(p) = clip(E(p) * base^(B(p) - 1), 0, 1)
M(p)  = 1[E'(p) > tau]
```

where `v_k` is the crop validity mask (0 in padded regions).

Precise deltas, each justified:

| Aspect | masker | BAM | Why |
|---|---|---|---|
| Weighting | uniform average | uniform by default, `gaussian` optional | Parity first; Gaussian center-weighting is ablated in M3 |
| `sensitivity` | 0-100 int mapped to `base^(n-1)` | float 1.0-3.0, default **1.0** | `base=5` with 2-3 overlaps saturates to 1.0 and makes the threshold meaningless |
| Denoise | DBSCAN on pixel coords | connected-component area filter default, DBSCAN optional | 10-20x faster, equivalent for thin structures |
| Morphology | close(2) + open(1) + median | same, defaults re-tuned | Preserved |
| Canvas dims | taken from first crop | validated against the request, mismatch is a 400 | Silent inconsistency is a bug |

### 5.4 Edge to mask (new)

The model predicts an edge; the product needs a filled object mask. This module is deterministic and runs after the merge.

```
Inputs:  E (float HxW edge probability), S (uint8 stroke raster), params
Outputs: mask (uint8 HxW), confidence (0-1), mode_used
```

1. **Smooth and binarize.** `E_s = gaussian(E, 0.8)`, `Me = E_s > tau_edge` (default 0.45).
2. **Thin.** `skeletonize(Me)`, then `close(disk(1.5))` to bridge 1-2 px gaps.
3. **Barrier.** `barrier = dilate(Me, disk(1))`. A 1 px edge map leaks; a 3 px barrier does not.
4. **Loop test.** Fill the stroke raster: `loop = binary_fill_holes(S)`. Valid if `0.005 < area(loop)/(H*W) < 0.95`. Otherwise go to step 8.
5. **Outside flood.** Add a 1 px frame to `barrier` (handles objects touching the border), then flood fill from all four corners through `~barrier`. `outside` = reached region.
6. **Mask.** `mask = ~outside`. Intersect with `dilate(loop, 8)` to reject unrelated closed regions elsewhere in the frame. Keep components whose area >= 0.2% of the frame. Holes are preserved automatically, because an enclosed region is not reachable from the corners.
7. **Refine.** `guidedFilter(guide=RGB, radius=8, eps=1e-3)` on the float mask, threshold 0.5, then `expand` px of dilate/erode as requested. Smooth the contour with `approxPolyDP(epsilon=0.8)`.
8. **Fallbacks, in order:**
   - a. GrabCut with `inside_seed = erode(loop, 3)` as sure-foreground, `~dilate(loop, 8)` as sure-background, `barrier` as probable-background.
   - b. `loop` refined by the guided filter (rough but never empty).
   - c. Edge-ribbon mode: `dilate(Me, r)`, for cracks and thin structures. This is the masker behavior, kept as an explicit mode.
9. **Confidence.** `1 - |area(mask) - area(loop)| / max(area(loop), 1)`, clipped to [0,1]. Below 0.4, the response sets `low_confidence: true` and the UI suggests closing the loop.

Multi-object: each connected component of `loop` is processed independently and the masks are unioned. This is how a user segments two objects with two loops in one stroke pass.

`mode: "object" | "line"` is exposed in the UI. `"line"` skips steps 4-8 and returns the edge ribbon, which preserves the original masker use case for cracks.

### 5.5 Parameters exposed to the user

| UI control | Range | Default | Maps to |
|---|---|---|---|
| Sensitivity | 1.0-3.0 | 1.0 | `brush_multiplier_base` |
| Edge threshold | 0.2-0.8 | 0.45 | `tau_edge` |
| Mask threshold | 0.2-0.8 | 0.5 | `tau` |
| Expand / contract | -8..+8 px | 0 | Post-fill morphology |
| Smoothing | 0-4 | 2 | Guided filter radius scale |
| Mode | object / line | object | Fill strategy |
| Show edge | toggle | off | Overlay the raw edge heatmap |

### 5.6 Failure modes and handling

| Failure | Detection | Response |
|---|---|---|
| No crops processed | `processed == 0` | 500 + toast "Model unavailable" |
| Empty prediction | `max(E) < 0.1` | Return empty mask, hint "try a thicker stroke" |
| Open loop | Loop test fails | Fallback chain, `low_confidence: true` |
| Fill leaked to whole frame | `area(mask) > 0.9*H*W` | Reject, fall back to loop mask |
| Crop coordinates out of bounds | Validation | Clamp, log, exclude from merge |
| Payload too large | `len(crops) > 64` or > 24 MB | 413 with a clear message |

### 5.7 Latency budget

Target hardware: 8-core desktop CPU, no GPU.

| Stage | fp32 torch | int8 onnx |
|---|---|---|
| Crop encode (browser) | 20-40 ms | 20-40 ms |
| Network transfer (localhost) | 10-20 ms | 10-20 ms |
| 24 crops forward (batch) | 750-950 ms | 350-480 ms |
| Merge + edge->mask + PNG | 60-110 ms | 60-110 ms |
| **Total, 24 crops** | **~0.9-1.2 s** | **~0.45-0.65 s** |
| **Total, 8 crops (typical)** | **~0.4 s** | **~0.2 s** |

If p95 exceeds 1.5 s, the mitigations in order are: INT8 default, stride 160 instead of 128 (fewer crops), width 16, and a "fast preview" mode that predicts on a 2x-downscaled canvas during the drag and re-runs at full quality on mouseup.

---

## 6. Backend

### 6.1 Stack

- Python 3.11, Flask 3, PyTorch 2.x (CPU wheel), ONNX Runtime, OpenCV (headless), NumPy, scikit-image, PyYAML.
- No database. Datasets are directories; state is the filesystem.
- Dev: `python run.py`. Local production: `waitress-serve --threads=8 --port=8000 backend.app:create_app`.

### 6.2 Module layout

| Module | Responsibility |
|---|---|
| `app.py` | App factory, blueprint registration, error handlers, request size limits |
| `routes/pages.py` | `/`, `/datasets`, `/masking` |
| `routes/datasets.py` | List datasets/images, serve media files |
| `routes/masks.py` | Load/save masks, upload image |
| `routes/magic_pen.py` | `/magic_pen/predict_crops` (compat) and `/api/segment` |
| `routes/health.py` | `/api/health`, `/api/warmup`, `/api/model_info` |
| `inference/registry.py` | Singleton model holder, backend selection, warmup, thread lock |
| `inference/predictor.py` | Preprocess, batch forward, sigmoid, return float map |
| `inference/merger.py` | `PredictionMerger`, preserved semantics |
| `inference/edge_to_mask.py` | The fill pipeline (§5.4) |
| `services/storage.py` | Safe path resolution, atomic writes, base64 decode/encode |
| `services/dataset_index.py` | Scans `datasets/`, caches counts and thumbnails |
| `services/cache.py` | LRU on `(image_id, crop_hash, params_hash)` |

### 6.3 API

Full schemas in Appendix B. Summary:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{status, device, backend, model, version}` |
| GET | `/api/datasets` | Dataset list with image counts |
| GET | `/api/datasets/<ds>/images` | Filenames, paginated |
| GET | `/media/<ds>/<split>/<name>` | Serve image with cache headers |
| POST | `/api/mask/save` | Save mask PNG |
| POST | `/api/image/upload` | Upload image into a dataset |
| POST | `/magic_pen/predict_crops` | **Compatible** with masker's contract |
| POST | `/api/segment` | Extended: edge + filled mask + stats |
| POST | `/api/warmup` | Pre-load the model |

`/magic_pen/predict_crops` keeps the masker request/response shape so the original frontend keeps working during migration. `/api/segment` is the new path used by the redesigned frontend.

### 6.4 Model service

- Loaded once at startup into a module-level singleton. The masker instantiates `Predictor` twice (`app.py` and `magic_pen_router.py`), doubling memory and load time; BAM uses a registry.
- Device from `config.infer.device` (`cpu` default, `cuda` optional). Never auto-selects CUDA for inference, because the product promise is CPU.
- Warmup: 3 dummy forwards at startup, so the first real request does not pay one-time kernel costs.
- Batching: all crops in one request are stacked into a single `[N,4,256,256]` tensor. This is the single biggest latency win versus per-crop loops.
- Thread safety: one `threading.Lock` around forward. CPU forward is not reliably reentrant across threads, and a lock is cheaper than oversubscription.
- Cache: `blake2b(crop_bytes + params)` -> probability map, LRU 2,048 entries. Undo/redo and repeated strokes become near-instant.

### 6.5 Storage

- Paths resolved with `Path.resolve()` and checked against the dataset root; reject traversal.
- Atomic writes: write to `.tmp`, `os.replace`.
- Masks saved as PNG, 8-bit, 0/255, same stem as the image.
- Never `path.split('.')[0]` to build filenames — that truncates names containing dots (a real bug in masker).

### 6.6 Config

`config.py` defines dataclasses; `config.yaml` holds defaults; environment variables (`BAM_*`) and CLI flags override. Precedence: CLI > env > YAML > dataclass defaults. The fully resolved config is written into every checkpoint and exposed at `/api/model_info`.

See Appendix A for the full schema.

### 6.7 Security and limits

- `MAX_CONTENT_LENGTH = 24 MB`.
- Max 64 crops per request; max 4,096 px on any side for uploads.
- Reject non-image payloads by magic bytes, not by extension.
- Bind to `127.0.0.1` by default. `0.0.0.0` requires an explicit flag and a warning.
- `debug=False` in any non-dev run.
- No shell execution, no user-controlled paths outside the dataset root.

---

## 7. Frontend

### 7.1 Stack and constraints

Keep the masker stack: Jinja2 templates, vanilla ES modules, Canvas 2D. No framework, no build step. This keeps the app trivially runnable and matches the existing logic. The `Canvas` class API (`drawImage`, `drawMask`, `getMaskBase64`, `switchBrushMode`, `zoomIn`, ...) is preserved so `ui.js` and `file_system.js` keep working.

### 7.2 Layout

```
┌────────────────────────────────────────────────────────────────────┐
│ BAM   dataset ▾   image 12 / 240        [Export ▾]   [Theme]  [?]  │
├──────┬──────────────────────────────────────────┬──────────────────┤
│  B   │                                          │  Magic Pen       │
│  E   │                                          │   Sensitivity  ──│
│  M   │              canvas                      │   Edge thr.    ──│
│  F   │         (checkerboard void,              │   Mask thr.    ──│
│  ⌘   │          image centered)                 │   Expand       ──│
│      │                                          │  ─────────────   │
│ ──── │                                          │  Display         │
│  ↶   │                                          │   Mask  Edge     │
│  ↷   │                                          │  ─────────────   │
│      │                                          │  Model           │
│      │                                          │   bam_v1 · CPU   │
├──────┴──────────────────────────────────────────┴──────────────────┤
│ 1024x768 · 67% · 14 crops · 312 ms · int8                          │
└────────────────────────────────────────────────────────────────────┘
```

- Left rail: tools and history. 48 px, icon buttons with tooltips.
- Center: canvas on a neutral checkerboard so transparent regions read correctly.
- Right: inspector, 280 px, grouped, collapsible, values persisted to `localStorage`.
- Status bar: always-visible, monospace numbers. This is the "instrument panel" that makes the tool feel professional.

### 7.3 Interaction

| Action | Input |
|---|---|
| Brush | `B`, drag |
| Eraser | `E`, drag |
| Magic pen | `M`, drag |
| Fill | `F` |
| Pan | `Space` + drag, or middle mouse |
| Zoom | wheel, `+` / `-`, `0` to fit, `1` to 100% |
| Brush size | `[` / `]` |
| Undo / redo | `Ctrl+Z` / `Ctrl+Shift+Z` |
| Save mask | `Ctrl+S` |
| Toggle mask | `Tab` |
| Toggle edge overlay | `Shift+Tab` |
| Subtract from mask | `Alt` + magic pen drag |
| Cancel in-flight request | `Esc` |

Cursor: a live circle at the true brush radius, with a crosshair at the center. In magic pen mode, additionally show the 256 crop square and a faint trail of emitted crop centers, so the user can see coverage. This is a small detail that makes the tool feel transparent rather than magic.

Feedback: during the drag, the stroke is drawn in amber. On mouseup, a thin progress bar appears in the status area with the crop count; the mask fades in over 120 ms. No blocking spinner. If inference exceeds 400 ms, show "snapping 14 crops...".

### 7.4 Visual system

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#F6F7F9` | `#0F1115` | App background |
| `--panel` | `#FFFFFF` | `#171A21` | Panels, rails |
| `--border` | `#E3E6EB` | `#262B35` | Dividers |
| `--text` | `#14171C` | `#E8EBF0` | Primary text |
| `--muted` | `#5B6472` | `#98A2B3` | Secondary text |
| `--accent` | `#4F46E5` | `#818CF8` | Primary action, mask contour |
| `--stroke` | `#F59E0B` | `#FBBF24` | Magic pen stroke |
| `--mask-fill` | `rgba(79,70,229,.32)` | `rgba(129,140,248,.32)` | Mask overlay |
| `--canvas-void` | `#E9ECF1` | `#0A0C10` | Area outside the image |
| `--ok` / `--warn` / `--err` | `#059669` / `#D97706` / `#DC2626` | lighter variants | Status |

Type: Inter (UI), JetBrains Mono (numeric readouts). Scale 12/13/14/16/20. Spacing on an 8 px grid. Radius 6 px for controls, 10 px for panels. One shadow level only (`0 1px 2px rgba(0,0,0,.06)`), used on floating elements.

The mask overlay changes from masker's flat 50% red to an indigo fill with a 1 px contour. Red at 50% hides the very boundary the user is trying to judge.

### 7.5 Performance

- Crop extraction uses `drawImage(imgCanvas, sx, sy, 256, 256, 0, 0, 256, 256)` on a reused offscreen canvas. The masker uses `getImageData` + `putImageData` per crop, which is 5-10x slower and allocates heavily.
- One `AbortController` per stroke; a new stroke cancels the previous request.
- Mask preview is composited at 50% scale during interaction, full scale on idle.
- Thumbnails in the dataset picker are lazy-loaded and cached by the browser.
- Avoid layout thrash: canvas size is set once per image, not per frame.

### 7.6 Accessibility

- All tools reachable by keyboard, with visible focus rings.
- `aria-label` on every icon button; `aria-live="polite"` on the status bar.
- Color is never the only signal: the mask also has a contour, and low-confidence results carry a text hint.
- Minimum contrast 4.5:1 for text.
- `prefers-reduced-motion` disables the mask fade.
- Works at 1280x800 and above; panels collapse below 1100 px.

### 7.7 Improvements over masker

| Area | masker | BAM |
|---|---|---|
| Toolbar | 14 stacked emoji buttons | 48 px icon rail, grouped, tooltips, shortcuts |
| Settings | fixed floating panel, `1vw` fonts | right inspector, readable, persisted, grouped |
| Image loading | full image as base64 in JSON | URL + HTTP cache, base64 only as fallback |
| Crop extraction | `getImageData` per crop | `drawImage` from a cached canvas |
| Payload | PNG per crop | WebP q90 + one stroke raster |
| Feedback | `console.log` | status bar, progress, toasts |
| Mask display | flat 50% red | fill + contour + optional edge heatmap |
| Export | save into dataset only | mask PNG, cutout PNG, polygon JSON, clipboard |
| Undo | single-level | 30-step history with labels |
| Zoom | buttons only | wheel, fit, 100%, pan |
| Errors | console | inline toast with retry |
| Model control | none | device/backend shown, warmup, latency |

---

## 8. Evaluation

### 8.1 Splits

| Split | Size | Composition |
|---|---|---|
| Train | 80% of Tier A | Open Images + COCO, deduped |
| Val | 10% of Tier A | Used for model selection only |
| Test-A | 10% of Tier A | Never tuned on; the shipping metric |
| Test-B | DIS5K VD/TE + DUTS-TE + ECSSD + HKU-IS + DUT-OMRON | Research comparability |
| Test-H | 200 in-house human strokes | Real-vs-synthetic calibration |

Test-A and Test-B are frozen with a hash. Any change requires a new version number.

### 8.2 Metrics

**Boundary quality (model output, before fill):**
- `BF@2px` — boundary F-measure with 2 px tolerance. Primary selection metric.
- `Boundary IoU` — IoU of boundary bands, sensitive to localization.
- `MAE` on the soft band.

**Mask quality (after edge->mask):**
- `IoU`, `Boundary IoU`, `S-measure`, `E-measure`, `F-measure`.
- `Hole accuracy`: IoU restricted to annotated holes.

**Product metrics:**
- `IoU@1stroke`, `IoU@2strokes`.
- `Success@0.9`: fraction of images reaching IoU >= 0.9 with one stroke.
- `Stroke length to 0.9`: median px of stroke needed.
- `Low-confidence rate`: fraction of requests with `confidence < 0.4`.

**System metrics:**
- p50/p95 latency per crop and per stroke, fp32 and int8.
- Peak RSS.
- Model size.

### 8.3 Targets

These are targets to validate in M2/M3, not guarantees.

| Metric | Target | Stretch |
|---|---|---|
| BF@2px (Test-A) | 0.80 | 0.84 |
| Mask IoU (Test-A) | 0.85 mean | 0.88 |
| Mask IoU (Test-B, DIS5K) | 0.80 | 0.84 |
| Success@0.9 (easy+medium) | 70% | 80% |
| p95 latency per crop, fp32 | <= 60 ms | 40 ms |
| p95 latency per crop, int8 | <= 30 ms | 20 ms |
| Model size, int8 | <= 1 MB | 0.5 MB |
| Peak RSS | <= 1.2 GB | 800 MB |

### 8.4 Protocol

1. Run `training/evaluate.py --split test_a --checkpoint model/bam_v1.pth`.
2. Strokes for evaluation are generated with a **held-out seed range** (10,000-19,999) never used in training, so the model cannot memorize a specific jitter.
3. Report mean and median; boundary metrics are heavy-tailed.
4. Slice by object size (`<32²`, `32-96²`, `96-256²`, `>256²`), by category group, and by stroke offset bucket.
5. Every reported number is produced by a script, never by hand. Results land in `reports/eval_<date>.json` and are rendered to a Markdown table.

### 8.5 Ablations

| ID | Variable | Values | Question |
|---|---|---|---|
| A1 | Stroke encoding | binary / SDF | Does an explicit distance signal help? |
| A2 | Crop mixture | with / without empty and clutter buckets | Does the anti-shortcut mixture matter? |
| A3 | Stroke realism | jitter only / full generator | How much does wobble + coverage matter? |
| A4 | Target | 1 px line / sigma 1.0 band / sigma 1.5 band | Band width vs precision |
| A5 | Block | DSConv / vanilla | Accuracy cost of the CPU-friendly block |
| A6 | Width | 16 / 20 / 24 / 32 | Accuracy vs latency frontier |
| A7 | Merge weighting | uniform / gaussian | Does center-weighting reduce edge blur? |
| A8 | Fill | flood / grabcut / loop-refine | Which fallback actually fires, and how often |
| A9 | Quality weighting | on / off | Does label-quality weighting sharpen boundaries? |
| A10 | Pretraining | scratch / ImageNet encoder | Is pretraining worth the license dependency? |

Each ablation is a single-variable change against the M3 baseline, same seed, same schedule. Results table goes in `reports/ablations.md`.

### 8.6 Human stroke study

Synthetic strokes are a proxy. To measure the gap:

1. Recruit 3-5 people, give them the app with the model disabled (draw mode only), and ask them to outline 40 objects each.
2. Record real stroke rasters and timing.
3. Compare distributions: brush radius, wobble amplitude, coverage, offset from GT, drawing speed.
4. Re-run Test-H with real strokes. If `IoU@1stroke` drops more than 5 points versus synthetic, the generator is mis-calibrated — adjust the mixture and fine-tune for 5 epochs on a small real-stroke set.

This is the single highest-value evaluation in the plan, because it measures the actual deployment distribution rather than a simulation of it.

---

## 9. Training operations

### 9.1 Hardware

| Role | Spec | Notes |
|---|---|---|
| Training | 24 GB GPU | Batch 32 at 256² uses ~4-6 GB. Plenty of headroom. |
| Training fallback | 16-32 GB RAM CPU | ~0.5-1.5 it/s. 60 epochs becomes 10-25 h. Viable overnight, not for iteration. |
| Inference | 8-core CPU | The product target. Must be benchmarked on the target machine, not a dev laptop. |

Config: `train.device: "auto"`, `infer.device: "cpu"`. AMP on by default when CUDA is present; disabled on CPU.

### 9.2 Time estimate

| Stage | GPU | CPU |
|---|---|---|
| Download + normalize 100K images | 2-4 h (I/O bound) | same |
| Stroke synthesis (100K) | 20-40 min | same |
| Overfit sanity check | 10 min | 40 min |
| 5K-image pilot, 10 epochs | 15 min | 3 h |
| Full run, 60 epochs | 1.5-3 h | 10-25 h |
| Evaluation (Test-A + Test-B) | 20 min | 1.5 h |
| ONNX export + parity | 10 min | 10 min |

### 9.3 Checkpoints

```python
{
  "model_state": ...,
  "config": {...},            # fully resolved
  "manifest_hash": "...",
  "data_tier": "A",           # license audit trail
  "epoch": 42,
  "metrics": {"val_bf2": 0.812, ...},
  "git_commit": "...",
  "created_at": "2026-11-11T14:00:00+08:00"
}
```

Only checkpoints with `data_tier` and `manifest_hash` may be loaded by the backend. This prevents accidentally shipping a research-tier model.

### 9.4 Reproducibility

- Global seed for Python, NumPy, PyTorch.
- Deterministic dataloader with a per-worker seed derived from `(epoch, worker_id)`.
- `torch.backends.cudnn.benchmark = False` during training for run-to-run comparability (slower, but comparable).
- Every run writes `runs/<timestamp>/{config.yaml, metrics.csv, git_commit.txt}`.
- A run is reproducible from `config.yaml` + `manifest_hash` alone.

---

## 10. Milestones

| ID | Deliverable | Acceptance criteria | Est. |
|---|---|---|---|
| **M0** | Scaffold | Repo layout, `config.py` with YAML/env/CLI merge, `training/unet.py` importable standalone, CI runs lint + unit tests | 0.5 d |
| **M1** | Data pipeline | `prepare/` produces canonical `rgb/gt/gt_edge` for Tier A; `verify_alignment.py` passes on 100% of samples; 200-sample contact sheet reviewed | 2 d |
| **M2** | Stroke generator + pilot model | Invariant tests pass; overfit-32 converges; 5K pilot beats the copy-stroke baseline; width/latency table measured | 2 d |
| **M3** | Full training + eval | Test-A targets met or documented; ablations A1-A7 run; ONNX + INT8 parity verified; `reports/eval_*.json` | 3 d |
| **M4** | Backend | All endpoints; `/magic_pen/predict_crops` compatible; merge parity test vs masker on a fixed crop set; edge->mask unit tests; cache and batching | 2 d |
| **M5** | Frontend | Redesigned UI per §7; magic pen end-to-end; keyboard, zoom, undo, export; no console errors; 1280x800 layout verified | 3 d |
| **M6** | Packaging | `README.md`, license matrix, model card, `requirements.txt`, one-command demo with 20 sample images | 1 d |
| **M7** | Optional | Multi-object, high-res mode (1024), folder batch mode, polygon export | 2 d |

Total: ~13.5 person-days to M6. M1 and M3 are the schedule risks, both I/O and GPU bound.

**Definition of done for v1:** a user can open the app, load any image, draw one rough loop around an object, and get a mask with IoU >= 0.85 in under 1.2 s on an 8-core CPU with no GPU installed.

---

## 11. Risks

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | **Identity shortcut** — model copies the stroke channel | High | High | Stroke jitter/offset, 8% dropout, distractor strokes, thin soft target, overfit test that checks output != input |
| R2 | **Synthetic-to-real gap** — real strokes differ from simulation | High | Medium | Test-H human study (§8.6), 5-epoch fine-tune on real strokes, parameter calibration |
| R3 | **Coarse GT** (DUTS, some Open Images) teaches blur | High | High | Quality weights, boundary sharpness filter, per-dataset target bandwidth |
| R4 | **Scale mismatch** at inference | Medium | Medium | Canonical inference scale, scale augmentation, sliced eval by object size |
| R5 | **Fill leakage** through edge gaps | High | Medium | 3 px barrier, multi-strategy fallback, confidence score, UI hint |
| R6 | **CPU latency** above budget | Medium | Medium | DSConv, width 20, batching, INT8, stride reduction, fast preview |
| R7 | **License contamination** | High (legal) | Medium | Tier A/B split, `data_tier` in checkpoint, backend refuses non-A checkpoints, README matrix |
| R8 | **Thin / transparent / occluded objects** fail | Medium | High | Documented limits, `line` mode for thin structures, UI guidance |
| R9 | **Class imbalance** | Medium | Medium | Soft band target, `pos_weight=6`, Dice |
| R10 | **Single-object bias** — model expects one centered object | Medium | Medium | Multi-instance crops from COCO/Open Images, dense/clutter bucket |
| R11 | **Dataset drift** — manifest changes between runs | Medium | Low | Manifest hash in checkpoint, immutable test splits |
| R12 | **Windows path and encoding issues** | Low | Medium | `pathlib` everywhere, ASCII filenames, no shell interpolation |

---

## 12. Decisions log

| # | Decision | Alternative rejected | Why |
|---|---|---|---|
| D1 | 4-channel input, 1-channel soft edge output | Predict mask directly | Matches the brief; avoids the inside/outside sign ambiguity; matches masker's edge logic |
| D2 | Fill is a separate deterministic stage | Second network head for interior | Keeps the model tiny and debuggable; fill failures are explainable |
| D3 | Canonical short side 640 | Match lowest dataset (~400) | Preserves detail and keeps crop content local, matching inference |
| D4 | Canonical inference space | Native-resolution crops | Eliminates scale mismatch and bounds CPU cost |
| D5 | Depthwise-separable UNet, width 20 | Vanilla UNet width 64 | ~30x fewer FLOPs; the task is local and low-level |
| D6 | Soft boundary band target | 1 px line | Trainable under class imbalance; tolerant of 1 px annotation noise |
| D7 | Online stroke synthesis | On-disk `gt_edge` only | Unbounded augmentation; `gt_edge/` kept for QA and reproducibility |
| D8 | Flood-fill fill with fallbacks | GrabCut as primary | Faster, deterministic, preserves holes; GrabCut kept as fallback |
| D9 | CC area filter default, DBSCAN optional | DBSCAN only | Equivalent for thin structures, 10-20x faster |
| D10 | Tier A/B data split | Train on everything | Legal safety without losing benchmark comparability |
| D11 | Keep Flask + vanilla JS | React/Vue rewrite | Preserves masker logic and zero-build simplicity |
| D12 | Uniform merge default | Gaussian merge default | Parity with masker; Gaussian is an ablated option |

---

## 13. Open questions

1. **High-res mode.** Is 640 short side enough for the intended objects, or is a 1024 working scale needed by default? Depends on target imagery. Recommend deciding after M2 visual QA.
2. **SDF stroke encoding.** If A1 shows a clear win, the "RGB + edge" description should be updated to "RGB + stroke distance field" — still 4 channels, different encoding.
3. **Multi-object UX.** Two loops in one stroke versus two separate strokes. Current plan supports both; needs a UX decision.
4. **Export formats.** Polygon JSON is listed; is COCO polygon export needed for downstream labeling pipelines?
5. **Batch mode.** Folder-in / folder-out segmentation is a common request for a tool like this, but it changes the UI substantially. Deferred to M7.
6. **Model distribution.** Ship the `.pth` in-repo (1.3 MB, simple) or via release download? In-repo is simpler and the file is small.

---

## Appendix A — Config reference

```yaml
seed: 1337

paths:
  datasets: ./datasets
  model: ./model/bam_v1.pth
  runs: ./runs

data:
  tier: "A"                  # A = shippable, B = research
  canonical_short_side: 640
  crop_size: 256
  stride: 128
  stroke_source: "online"    # online | disk
  stroke_encoding: "binary"  # binary | sdf
  stroke:
    radius_min: 1.5
    radius_max: 6.0
    jitter_sigma: 3.0
    wobble_amp: [1.0, 4.0]
    wobble_len: [20, 180]
    coverage: [0.45, 1.00]
    p_no_stroke: 0.08
    p_distractor: 0.15
    p_interior: 0.05
  crop_mix:
    boundary: 0.50
    near: 0.20
    empty: 0.15
    dense: 0.15
  target_sigma: 1.0
  num_workers: 8

model:
  in_channels: 4
  out_channels: 1
  width: 20
  depth: 4
  block: "ds"                # ds | vanilla
  bilinear: true

train:
  device: "auto"
  batch_size: 32
  crops_per_epoch: 24000
  epochs: 60
  lr: 3.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 3
  amp: true
  ema_decay: 0.999
  grad_clip: 1.0
  val_crops: 2000
  early_stop_patience: 15

infer:
  device: "cpu"              # cpu | cuda
  backend: "auto"            # auto | onnx_int8 | onnx_fp32 | torch
  working_short_side: 640
  crop_size: 256
  stride: 128
  max_crops: 64
  batch_size: 16
  threads: 8
  threshold: 0.5
  edge_threshold: 0.45
  sensitivity: 1.0
  expand: 0
  smooth: 2
  mode: "object"             # object | line
  cache_entries: 2048

server:
  host: "127.0.0.1"
  port: 8000
  max_content_mb: 24
  debug: false
```

---

## Appendix B — API schemas

### `POST /api/segment`

```json
{
  "dataset": "openimages",
  "image": "000012345",
  "canvas": {"width": 853, "height": 640},
  "stroke_mask": "data:image/png;base64,...",
  "stroke_path": [[120, 300], [180, 290]],
  "crops": [
    {
      "id": 0,
      "x0": 64, "y0": 128,
      "centerX": 192, "centerY": 256,
      "width": 256, "height": 256,
      "image_base64": "data:image/webp;base64,..."
    }
  ],
  "params": {
    "sensitivity": 1.0,
    "threshold": 0.5,
    "edge_threshold": 0.45,
    "expand": 0,
    "smooth": 2,
    "mode": "object"
  }
}
```

Response:

```json
{
  "status": "success",
  "mask_base64": "data:image/png;base64,...",
  "edge_base64": "data:image/png;base64,...",
  "confidence": 0.87,
  "low_confidence": false,
  "mode_used": "flood",
  "stats": {
    "num_crops": 14,
    "num_failed": 0,
    "ms_infer": 412,
    "ms_post": 71,
    "backend": "onnx_int8"
  }
}
```

### `POST /magic_pen/predict_crops` (masker-compatible)

Same request as masker (`crops[]`, `mode`, `sensitivity`, `threshold`, `apply_morphology`, `apply_dbscan`, `db_eps`, `db_min_samples`). Response keeps `merged_mask_base64`, `num_crops_processed`, `canvas_dimensions`. Additional fields are additive only. This endpoint returns the **edge** mask, not a filled mask, preserving masker semantics.

### `GET /api/health`

```json
{
  "status": "ok",
  "device": "cpu",
  "backend": "onnx_int8",
  "model": "bam_v1",
  "data_tier": "A",
  "version": "1.0.0"
}
```

---

## Appendix C — Findings in masker to fix

Observed directly in the reference source. Each is fixed in BAM, and each has a regression test.

| # | Location | Issue | Severity | Fix |
|---|---|---|---|---|
| 1 | `model/unet.py` | `from config import Config as cfg`, but `config.py` is absent from the repo; the model cannot be imported standalone | Blocker | Model takes constructor args; no global config import |
| 2 | `magic_pen_router.py` `predict_single_crop` | Uses `binary_mask` before assignment | Blocker | Assign from `prediction_result`, normalize to 0-255 |
| 3 | `predictor.py` `__main__` | Calls `dbscan(..., threshold=0.5)`, but the signature has no `threshold` | Blocker | Remove; add a real smoke test |
| 4 | `app.py` + `magic_pen_router.py` | `Predictor` instantiated twice, loading the model twice | High | Singleton registry |
| 5 | `canvas.js` `cropImageBase64` | Pads out-of-bounds crops with black, creating a fake strong edge | High | Replicate padding + validity mask |
| 6 | `app.py` `save()` | `path.split('.')[0] + postfix` truncates filenames containing dots | High | `Path.with_suffix()` |
| 7 | `ui.js` | `redo_btn` gets two click listeners, so redo fires twice | Medium | Single binding |
| 8 | `canvas.js` `handleMagicPenCropping` | Spacing keyed to `crop_every_len * crops.length`, fragile under retries | Medium | Fixed arc-length stride |
| 9 | `magic_pen_router.py` | `sensitivity` 0-100 mapped to `base^(n-1)`; with 2-3 overlaps it saturates and the threshold stops mattering | Medium | Float 1.0-3.0, default 1.0 |
| 10 | `magic_pen_router.py` | Canvas dimensions taken from the first crop only | Medium | Validate all crops against the request |
| 11 | `canvas.js` | `getImageData` + `putImageData` per crop | Medium | `drawImage` from a cached canvas |
| 12 | `app.py` | Full images returned as base64 JSON | Medium | URL + cache headers, base64 fallback |
| 13 | `magic_pen_router.py` | Debug `print(sensitivity)` left in | Low | Remove |
| 14 | `settings.js` | `1vw` font sizes; unreadable below ~1400 px width | Low | Fixed rem scale |
| 15 | `app.py` | `debug=True` in the main entry point | Low | Config-driven, default false |
| 16 | `app.py` | No payload size limit | Low | `MAX_CONTENT_LENGTH` |
| 17 | `canvas.js` | `crop_size` default 200 but `crop_every_len` default 200 vs `crop_size/2` after resize — inconsistent | Low | Single `stride` parameter |

---

## Appendix D — File naming and manifest

- One stem per sample: `000012345`.
- `rgb/<stem>.jpg` (JPEG q95), `gt/<stem>.png` (0/255), `gt_edge/<stem>.png` (0-255 alpha).
- Extensions may differ; the manifest is authoritative. Loaders must resolve through the manifest, never by string concatenation.
- Stems are ASCII, zero-padded to 9 digits, no spaces, no dots.
- `datasets/_qa/` holds contact sheets and rejection lists; gitignored.
- `datasets/manifest.json` is committed for the demo subset, gitignored for full corpora.

---

## Appendix E — Glossary

| Term | Meaning |
|---|---|
| **Stroke** | The rough line the user draws near the object boundary |
| **Edge GT (`gt_edge`)** | The synthetic stroke raster used as input channel 4 |
| **Boundary band** | The soft training target: blurred 1 px boundary |
| **Crop** | A 256x256 window centered on a point along the stroke |
| **Merge** | Accumulating per-crop predictions into a full-canvas probability map |
| **Fill** | Converting the merged edge map into a filled object mask |
| **Canonical scale** | Short side 640 px; the single coordinate space for training and inference |
| **Tier A / B** | Commercially shippable / research-only data |
| **BF@2px** | Boundary F-measure with 2 px tolerance |
| **Success@0.9** | Fraction of images reaching IoU >= 0.9 with one stroke |
