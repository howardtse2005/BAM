# BAM: Brush Anything Model (v1.1)

### Note
This is the early version 1.1 and still in progress. There might be rough edges or wrong predictions as the UNet-based BAM was only trained on DIS5K dataset which might be not enough for any random objects. BAM is lightweight and runs on CPU (although GPU is recommended for training).

## Demo

Demo 1:


https://github.com/user-attachments/assets/f8015672-6c44-4dc7-9f34-9b5ff878368e



Demo 2:



https://github.com/user-attachments/assets/95c235ee-d39b-4e27-ad70-e277e353d84d


## How to run

### 1. Create virtual environment

```bash
python -m venv .venv
```
For windows:
```bash
.venv\Scripts\activate
```
For macOS / Linux:
```bash
source .venv/bin/activate
```
Then, install requirements
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only machine
```

### 2. Download datasets

**DIS5K** is the dataset this demo is trained on — 5,470 images with
high-quality masks, in three predefined splits.

- Repo and instructions: <https://github.com/xuebinqin/DIS>
- Download: [Google Drive](https://drive.google.com/file/d/1O1eIuXX1hlGsV7qx4eSkjH231q7G1by1/view?usp=sharing)
  or [Baidu Pan](https://pan.baidu.com/s/1y6CQJYledfYyEO0C_Gejpw?pwd=rtgw) (code `rtgw`)

**Don't want to download?** A 24-image synthetic set is enough to exercise every
step below, and a prepared copy already sits in `datasets/demo` — skip to step 4
with `--set dataset=demo`. To rebuild it:

```bash
python tools/make_demo_subset.py --count 24                                    # -> datasets/_raw/demo
python -m training.prepare.preprocess --dataset datasets/_raw/demo --out datasets/demo
```

**Expected source layout.** A dataset is a folder with three splits. The split is
simply the folder you put an image in — there is no manifest and no auto-split.
Images and masks pair by filename stem, case-insensitively.

```
datasets/DIS5K/
├── train/
│   ├── im/     0001.jpg ...
│   └── gt/     0001.png ...     # single channel, object 255, background 0
├── val/
│   └── im/  gt/
└── test/
    └── im/  gt/
```

**Expected prepared layout after step 3.** Preprocessing writes a sibling folder
and adds the two generated rasters plus a report:

```
datasets/DIS5K_prepared/
├── train/
│   ├── im/         # downscaled image
│   ├── gt/         # mask, same size as im
│   ├── gt_edge/    # the clean line to predict   (the target)
│   └── gt_brush/   # the rough stroke to consume (the 4th input channel)
├── val/            # same four folders
├── test/           # same four folders
└── meta.json       # per-split counts, resizes and rejected images
```

`gt_edge` and `gt_brush` are never authored by hand — both are derived from `gt/`,
and both cover the same boundary, outer contour and holes included.

### 3. Preprocess the dataset and train the model (optional; pretrained model is available on `model/bam_v1.pth`

Preprocess the dataset.
```bash
python -m training.prepare.preprocess --dataset datasets/DIS5K
```

Train the model.

```bash
python -m training.train --set dataset=DIS5K_prepared
```
The best checkpoint is copied to
`model/bam_v1.pth`. Logs and the resolved config land in `runs/<timestamp>/`.

```bash
python -m training.train --dry-run                    # one synthetic batch, no dataset needed
python -m training.train --resume runs/<id>/last.pth
```

### 4. Test the model (optional)

```bash
python -m training.evaluate --split test
```

Prints IoU, boundary IoU, BF and MAE, and writes `runs/eval_test.json`. For faster
CPU inference, export ONNX:

```bash
python -m training.export_onnx --checkpoint model/bam_v1.pth --int8
```

### 5. Deploy the local website

```bash
python run.py                                     # http://127.0.0.1:8000
python run.py --host 0.0.0.0 --serve waitress     # LAN / production
```

With no checkpoint the app still starts: it falls back to a mock backend that
predicts a band around your stroke, and the inspector shows an amber warning.

## Configuration explanation

`config.py` is the only config

| Setting | Default | Meaning |
|---|---|---|
| `dataset` | `demo` | folder under `datasets_dir` to train on / serve |
| `datasets_dir` | `./datasets` | root of every dataset |
| `upload_dataset` | `uploads` | where the website stores uploaded images |
| `short_side` | `640` | preprocessing and canvas scale |
| `crop_size` | `256` | training crop, and the largest window the net runs at |
| `crop_min` | `64` | smallest inference window |
| `width` / `depth` | `20` / `4` | UNet size; 16 is faster, 32 more accurate |
| `batch_size` | `32` | training batch |
| `epochs` | `100` | upper bound; early stopping usually ends sooner |
| `lr` | `3e-4` | AdamW learning rate |
| `line_width` | `3` | thickness of the predicted line, canvas px |
| `fill_close` | `5` | hairline gap bridged before filling |
| `magic_brush_width` | `32` | visible brush width (the slider) |
| `magic_prompt_ratio` | `0.625` | thinner stroke actually fed to the model, clamped to 3–12 px |
| `backend` | `auto` | `onnx_int8` → `onnx_fp32` → `torch` → `mock` |
| `tier` | `A` | `A` = shippable, `B` = research only (use `B` for DIS5K) |
| `host` / `port` | `127.0.0.1` / `8000` | server bind |

The two brush widths are the point of the magic brush: you draw one 32 px stroke
and see only that, while the model is given a 12 px prompt inside it. The
difference is the tolerance the predicted line may move in, so a line a few
pixels off the centreline is still visibly your stroke.

## Website tools explanation

The site opens straight into the editor with an empty canvas. There is no dataset
browser: **Upload** your own image (stored under `datasets/uploads/`), and
**Open** lists what you have already uploaded.

| Tool | Key | What it does |
|---|---|---|
| Magic brush | `M` (default) | drag *roughly* along the edge; on release the clean line is committed to the mask |
| Fill | `F` | flood the region enclosed by the line; holes stay holes |
| Brush | `B` | paint the mask by hand |
| Eraser | `E` | erase from the mask |
| Pan | `H` or hold `Space` | move the canvas |
| Undo / redo | `Ctrl+Z` / `Ctrl+Shift+Z` | one stroke or one fill at a time |
| Zoom | `+` `-` `0` `1` | in, out, fit, 100% |

Typical pass: **Upload → drag the magic brush along the boundary → `F` → Save**
(`Ctrl+S`). `Alt`+drag subtracts instead of adding. Save writes
`datasets/uploads/train/gt/<name>.png` at the stored image's resolution and
refuses an empty mask.

The inspector holds four panels:

- **Magic brush** — brush width, sensitivity, edge threshold, fill threshold,
  expand (px), smoothing, merge weighting.
- **Brush** — size, show mask, show edge, mask opacity.
- **Model** — active backend, device, last stroke latency, data tier, warnings.
- **Image** — image size, mask state, last crop size, crop count, brush width.

A stroke is one forward pass: the server takes the stroke's bounding box, makes it
square, snaps it up to a multiple of 32 and runs the net once. Long strokes are
downscaled to `crop_size`, never upscaled, so the object scale matches training.

## Acknowledgement and License

**DIS5K** — Xuebin Qin, Hang Dai, Xiaobin Hu, Deng-Ping Fan, Ling Shao, Luc Van
Gool. *Highly Accurate Dichotomous Image Segmentation.* ECCV 2022.
<https://github.com/xuebinqin/DIS>

- The dataset ships under its own **Terms of Use**
  ([PDF](https://github.com/xuebinqin/DIS/blob/main/DIS5K-Dataset-Terms-of-Use.pdf)),
  a custom research licence rather than an open-source one. It grants use for
  academic research; commercial use is **not** granted, and the PDF sets out the
  redistribution terms. Read it before using the data.
- The images were collected from Flickr, so copyright in them stays with the
  original owners; the annotations belong to the DIS authors.
- The DIS5K code and evaluation metric are Apache-2.0. That covers the code, not
  the data.

**BAM itself** has no separate licence file yet. Until one is added, treat the
code as all rights reserved.
