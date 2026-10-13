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
- `GET /evaluate` for held-out validation accuracy, Dusty-positive precision,
  recall/F1, per-class and macro metrics, and a row-actual / column-predicted
  confusion matrix
- `GET /health`

The authenticated dashboard provides image and video analysis, side-by-side
original/processed comparisons, dust coverage and classification, model
confidence, and maintenance recommendations. Supported image formats are PNG,
JPG/JPEG, and WebP; video uploads must be MP4. After analysis, download an HTML
report with the results and media comparisons. Image report media is embedded;
video reports link to the uploaded and processed videos.

The dashboard's validation scores and score chart are measured on the
deterministic, stratified 20% held-out split of `dataset/Clean` and
`dataset/Dusty` for the currently loaded checkpoint. Accuracy uses all held-out
predictions. The headline precision, recall, and F1 treat Dusty as the positive
class and are calculated directly from TP, FP, and FN in the confusion matrix;
macro-averaged metrics and per-class metrics are also returned separately. The
confusion matrix shows actual labels as rows and predicted labels as columns.
This validation split is for model evaluation, not a substitute for a separate
external test set.

## Retrain from a dataset

```bash
cd corrected
python train.py --data-dir dataset --output model_mobilenetv2_trained.pth
```

The flat `dataset/Clean` and `dataset/Dusty` folders are split reproducibly
into training, validation, and held-out test samples. Training initializes the
MobileNetV2 feature extractor with ImageNet weights. The resulting checkpoint
does not replace the currently configured `model.pth`; point `MODEL_PATH` at it
to use it with `app.py`. Other folders in the dataset (for example Bird-drop,
Electrical-damage, Physical-Damage, or Snow-Covered) are ignored by the
binary Clean/Dusty panel-condition classifier and are not included in its
reported metrics. Training saves a training-versus-validation cross-entropy
loss graph, a held-out test confusion-matrix graph, and a JSON metrics report
next to the checkpoint. The JSON includes Clean and Dusty class precision,
recall, F1, and support, along with Dusty-positive precision/recall/F1,
accuracy, and the actual-row/predicted-column confusion matrix. The test set
is kept out of training and validation and is evaluated after selecting the
checkpoint with the best validation accuracy.

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

Efficiency loss prefers a measurement when the server has these physical
readings configured:

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

Without all four readings, the dashboard uses a Random Forest trained on
synthetic samples generated from the provisional assumption
`estimated loss (%) = 0.2 x dust coverage (%)`. The result is labeled as a
synthetic estimate, not as a physical measurement or a validated prediction.
Its accuracy must not be assumed until calibrated and evaluated against
real-world clean/dusty panel power and irradiance measurements. The estimate
is capped at 20% for the 0-100% coverage range.

The old `dashboard_*.py` file is JavaScript, not Python, and also depends on an
HTML template that was not included in the upload. It should be renamed to
`dashboard.js` if it is retained as a frontend script.
