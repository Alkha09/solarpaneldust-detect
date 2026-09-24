# SolarSense corrected inference package

## What was corrected

The supplied files mixed two incompatible pipelines:

- `train_*.py` and `app_*.py` expected a TensorFlow EfficientNet model with a
  one-value sigmoid output.
- `model_*.pth` is a PyTorch MobileNetV2 checkpoint with a
  `1280 -> 256 -> 2` classifier head.

This package uses the supplied PyTorch checkpoint consistently. It also fixes
the heatmap by explaining the `Dusty` two-class logit, applying the same
ImageNet normalization used by the classifier, masking coverage to the panel,
and leaving the non-panel background unchanged in the overlay. Rendered
heatmaps and processed videos are capped at 1920px on their largest side so
4K uploads do not exhaust memory.

## Run an image test

```bash
cd corrected
python test.py \
  --model ../attached_assets/model_1789918225387.pth \
  --image /path/to/panel-image.jpg \
  --output-dir test_output
```

The command writes `overlay.png` and `heatmap.png`.

## Run the API

```bash
cd corrected
MODEL_PATH=../attached_assets/model_1789918225387.pth python app.py
```

The API provides:

- `POST /predict` with multipart field `file`
- `POST /predict_video` with multipart field `video`
- `GET /video_status/<video_id>` for polling video progress
- `GET /health`

## Retrain from a dataset

```bash
cd corrected
python train.py --train-dir ../dataset/train --test-dir ../dataset/test
```

The resulting `model.pth` can be used directly by `app.py` and `test.py`.

## Train measured dust segmentation

Classification folders (`Clean` and `Dusty`) cannot provide measured pixel
coverage. For measured coverage, create paired annotations with this layout:

```text
segmentation/
  images/
    panel_001.jpg
  masks/
    panel_001.png
```

Each mask must be grayscale PNG: `255` for dust pixels and `0` for non-dust.
You can create masks interactively with:

```bash
python annotate_masks.py --images segmentation/images --masks segmentation/masks
```

Paint dust with the left mouse button, erase with the right button, press `S`
to save, and press `N` to skip.
Train and validate the U-Net with:

```bash
python train_segmentation.py --images segmentation/images --masks segmentation/masks --output segmentation_model.pth
```

Then start the app with `segmentation_model.pth` beside `app.py` (or set
`SEGMENTATION_MODEL_PATH`). The dashboard will report `U-Net pixel
segmentation`; without that checkpoint it reports `Grad-CAM estimate`.

Measured efficiency loss is calculated internally only when the server has
these physical readings configured; no fallback coefficient is used:

```text
Efficiency Loss (%) = (eta_clean - eta_dusty) / eta_clean * 100
eta = power (W) / irradiance (W/m2)
```

```powershell
$env:CLEAN_POWER_W = "1000"
$env:DUSTY_POWER_W = "800"
$env:CLEAN_IRRADIANCE_W_M2 = "1000"
$env:DUSTY_IRRADIANCE_W_M2 = "1000"
python app.py
```

Without all four readings, the dashboard shows `Not measured`.

The old `dashboard_*.py` file is JavaScript, not Python, and also depends on an
HTML template that was not included in the upload. It should be renamed to
`dashboard.js` if it is retained as a frontend script.
