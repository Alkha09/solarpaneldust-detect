#  Solar Panel Dust Detection System

**An AI-powered computer vision system that automatically detects dust accumulation on solar panels to support data-driven cleaning and maintenance decisions.**




##  Overview

Solar energy output degrades measurably when panel surfaces accumulate dust, sand, and airborne particulates — yet most installations still rely on manual, schedule-based inspection rather than condition-based monitoring. This project addresses that gap with a deep learning system that classifies panel images as **clean** or **dust-affected**, exposed through a lightweight web application for inference and visualization.

The system combines a trained Convolutional Neural Network (CNN) with an OpenCV-based preprocessing pipeline and a Flask web interface, allowing a user to upload a panel image and receive an instant classification with a confidence score. It is built as an end-to-end pipeline — covering dataset handling, model training, evaluation, and deployment — rather than a notebook-only experiment.

This repository demonstrates the practical application of computer vision to a renewable-energy operations problem, with attention to the full ML lifecycle: data preparation, model development, inference serving, and result interpretation.

---

##  Real-World Problem

Photovoltaic (PV) panels are typically installed outdoors and exposed to continuous environmental stress — dust storms, pollen, bird droppings, and general particulate buildup. This has measurable consequences:

- **Efficiency loss** — Dust accumulation can reduce panel energy output significantly depending on region, panel tilt, and dust density, directly impacting return on investment for solar installations.
- **Manual inspection is unscalable** — Large solar farms span hundreds of acres; physically inspecting every panel on a fixed schedule is labor-intensive, slow, and often reactive rather than predictive.
- **Unnecessary or delayed cleaning cycles** — Without data, operators either over-clean (wasting water and labor) or under-clean (losing energy yield), since there's no objective signal driving the decision.
- **Growing solar adoption** — As utility-scale and rooftop solar installations expand globally, manual monitoring approaches do not scale proportionally with deployment.

An automated, image-based detection system offers a low-cost, scalable alternative to manual inspection — turning a subjective visual check into a repeatable, data-backed classification task.

---

##  Proposed AI Solution

The system follows a straightforward, well-defined inference pipeline:

```
Input Image (Solar Panel Photo)
            ↓
Image Preprocessing (Resize, Normalize, Denoise — OpenCV)
            ↓
Deep Learning Model (CNN Classifier)
            ↓
Dust Classification (Clean / Dusty + Confidence Score)
            ↓
Result Visualization (Web Interface Output)
```

**Why this approach:** CNNs are well-suited to this problem because dust accumulation manifests as visual texture and color patterns (haze, grain, opacity) that are difficult to encode with hand-crafted rules but are learnable from labeled image examples. Framing this as an image classification problem keeps the model lightweight enough for practical deployment while still capturing the relevant visual signal.

---

##  Key Features

-  **Automated Dust Detection** — Removes the need for manual visual inspection of panels.
-  **Image-Based Analysis** — Accepts a panel photograph as the sole input; no special sensors required.
-  **AI Model Prediction** — CNN-based classifier outputs a clean/dusty prediction with an associated confidence score.
-  **Web Interface** — Flask-based UI for uploading images and viewing predictions without touching code.
-  **Result Visualization** — Displays the uploaded image alongside its predicted class for quick interpretation.
-  **Scalable Monitoring Concept** — Designed as a foundation that can extend to batch processing, multi-camera feeds, or fleet-wide panel monitoring.

---

##  System Architecture

The system is organized into three logical layers that work together end-to-end:

**1. Data & Training Layer**
Raw panel images are organized into labeled classes (clean / dusty), passed through preprocessing, and used to train the CNN. This layer is run offline, producing a saved model artifact.

**2. Inference Layer**
The trained model is loaded once at application startup. Incoming images are preprocessed using the same pipeline as training (consistency is critical for prediction reliability) and passed through the model to generate a prediction.

**3. Application Layer**
A Flask server exposes routes for image upload and result rendering. HTML/CSS templates handle the user-facing presentation, displaying the uploaded image and the model's prediction back to the user in real time.

```
┌──────────────────┐      ┌────────────────────┐      ┌─────────────────────┐
│   Training        │      │   Flask Backend     │      │   Web Frontend       │
│  (offline script) │ ───► │  (app.py + model)   │ ◄──► │ (templates/HTML/CSS) │
│  dataset/ →model   │      │  preprocessing +    │      │  upload + result      │
│                    │      │  prediction logic    │      │  display              │
└──────────────────┘      └────────────────────┘      └─────────────────────┘
```

---

## 🔬 Machine Learning Approach

**Dataset**
A labeled image dataset of solar panel surfaces, organized into two primary categories — *clean* and *dusty* — stored under `dataset/`. Images vary in lighting, angle, and dust density to encourage the model to generalize beyond a single capture condition.

**Data Preprocessing**
- Resizing all images to a fixed input dimension expected by the CNN.
- Pixel normalization to stabilize and speed up training convergence.
- Basic noise reduction / cleanup using OpenCV to reduce irrelevant visual artifacts.
- Train/validation split to monitor generalization during training rather than just training accuracy.

**Model Training**
A CNN architecture is trained on the preprocessed dataset to learn discriminative features between clean and dusty panel surfaces. Training is handled through a dedicated script, with the resulting model weights serialized for reuse — decoupling training from inference so the web app doesn't need to retrain on every run.

**Prediction Pipeline**
At inference time, a single uploaded image is run through the identical preprocessing steps used during training, passed through the loaded model, and the output is mapped to a human-readable label (Clean / Dusty) along with a confidence value.

**Evaluation**
Model performance is assessed on a held-out validation split using standard classification metrics (accuracy, and where applicable, precision/recall) to verify the model is learning meaningful patterns rather than overfitting to the training set. Results are reported honestly based on the dataset's size and diversity — performance on this dataset is a starting benchmark, not a claim of production-grade accuracy at scale.

---

## 🛠️ Tech Stack

**AI/ML:**
- Python
- TensorFlow / Keras (CNN model development)
- NumPy

**Computer Vision:**
- OpenCV (image preprocessing, resizing, transformations)

**Backend:**
- Flask (REST routes, model serving, request handling)

**Frontend:**
- HTML5, CSS3 (Jinja2 templates rendered via Flask)

**Tools:**
- Git & GitHub (version control)
- Jupyter / scripts for experimentation and training
- Virtualenv / pip (`requirements.txt`)

---

##  Repository Structure

```
solar-panel-dust-detection/
│
├── app.py                  # Flask application entry point — routes & inference logic
├── train_model.py          # Model training script
├── predict.py               # Standalone prediction/inference script
│
├── model/                   # Saved trained model file(s) (.h5 / .pkl)
│
├── dataset/                  # Labeled training images
│   ├── clean/
│   └── dusty/
│
├── templates/                # HTML templates for the web interface
│   ├── index.html
│   └── result.html
│
├── static/                   # CSS / uploaded image storage
│   └── images/
│
├── results/                   # Sample output screenshots
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/solar-panel-dust-detection.git
cd solar-panel-dust-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the application**
```bash
python app.py
```

The app will be available at `http://127.0.0.1:5000/` — upload a panel image through the web interface to receive a classification.

---

##  Results

Predictions are generated by passing an uploaded image through the preprocessing pipeline and the trained CNN, which outputs a class label (**Clean** / **Dusty**) along with a confidence score. The web interface displays the uploaded image side-by-side with the model's prediction for immediate interpretation.

| Input | Predicted Output |
|---|---|
| *[Screenshot: clean panel upload]* | `Clean — 9X.X% confidence` |
| *[Screenshot: dusty panel upload]* | `Dusty — 9X.X% confidence` |

> 📷 *Add screenshots to `results/` and reference them here, e.g.:*
> `![Web Interface](results/ui_demo.png)`
> `![Sample Prediction](results/prediction_sample.png)`

Quantitative evaluation (accuracy/precision/recall on the validation split) is logged during training and can be added here once finalized for this dataset.

---

##  Engineering Challenges

- **Image variation** — Differences in lighting, camera angle, panel orientation, and partial occlusion in real-world photos required careful preprocessing to keep predictions consistent across conditions.
- **Model accuracy improvement** — Iterating on CNN architecture, regularization, and training parameters to reduce overfitting on a relatively small, custom-curated dataset.
- **Dataset handling** — Sourcing, cleaning, and balancing a labeled clean/dusty image set, since public datasets for this specific use case are limited and class imbalance directly affects model reliability.
- **Deployment considerations** — Structuring the Flask app to load the trained model once at startup (rather than per-request) for reasonable inference latency, and managing file uploads/storage cleanly within the app's static directory.

---

##  Future Enhancements

-  **Real-time camera monitoring** — Continuous video feed analysis instead of single-image uploads.
-  **IoT integration** — Connecting edge cameras/sensors on physical panel arrays to feed live data into the model.
-  **Drone-based inspection** — Aerial imagery for large-scale solar farms, enabling coverage beyond fixed cameras.
-  **Cloud deployment** — Hosting the model and API on a cloud platform (AWS/GCP/Azure) for remote, multi-site access.
-  **Automated cleaning alerts** — Triggering maintenance notifications automatically once dust levels cross a defined threshold, closing the loop from detection to action.

---

##  Skills Demonstrated

- **Machine Learning** — End-to-end pipeline design: data preparation, training, evaluation.
- **Deep Learning** — CNN architecture design and training for image classification.
- **Computer Vision** — Image preprocessing and feature handling using OpenCV.
- **Python Development** — Modular, script-based engineering separating training, inference, and serving logic.
- **Flask Deployment** — Building and serving a functional ML-backed web application.
- **Problem Solving** — Translating an ambiguous real-world inefficiency (manual panel inspection) into a well-scoped, technically tractable ML problem.

---

## 👤 Author

**[Alkha]**

