"""Training package.

The model architecture lives here (``training/unet.py``) alongside the code that
trains it, rather than at the repository root: it is model code, not a runtime
entry point. ``model/`` holds weights only.

Modules
-------
unet.py         BAMUNet and build_model(); self-contained, no config import
stroke_synth.py synthetic user-stroke generation from a binary mask
sampler.py      crop sampling mixture (boundary / near / empty / dense)
dataset.py      torch Dataset, joint geometric and photometric augmentation
losses.py       BCE + soft Dice + L1
metrics.py      boundary F, boundary IoU, IoU, Dice, MAE
train.py        training loop, EMA, checkpointing
evaluate.py     checkpoint evaluation on a manifest split
export_onnx.py  ONNX export and INT8 quantisation
prepare/        download, normalise, split, dedupe

Nothing is re-exported here on purpose. The backend imports ``training.unet``
lazily, inside ``TorchBackend``, so the web app still starts on a machine without
PyTorch installed.
"""
