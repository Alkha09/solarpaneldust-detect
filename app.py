"""Flask API for SolarSense image and video dust analysis.

Run from this directory:
    MODEL_PATH=/path/to/model_1789918225387.pth python app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import uuid

import cv2
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from dust_level import analyze_dust, calculate_efficiency_loss
from gradcam import apply_dust_heatmap
from model import DEVICE, load_model, predict_rgb
from segmentation import load_segmentation_model, predict_dust_mask
from train import evaluate_flat_dataset
from video_processor import process_video


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "model.pth"))
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
VIDEO_UPLOAD_DIR = UPLOAD_DIR / "videos"
VIDEO_RESULT_DIR = RESULT_DIR / "videos"
for directory in (UPLOAD_DIR, RESULT_DIR, VIDEO_UPLOAD_DIR, VIDEO_RESULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "solar-sense-demo-secret")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
app.config["MODEL_PATH"] = str(MODEL_PATH)
model = load_model(MODEL_PATH)
SEGMENTATION_MODEL_PATH = Path(os.environ.get("SEGMENTATION_MODEL_PATH", BASE_DIR / "segmentation_model.pth"))
try:
    segmentation_model = load_segmentation_model(SEGMENTATION_MODEL_PATH) if SEGMENTATION_MODEL_PATH.exists() else None
except Exception:
    segmentation_model = None
HISTORY_FILE = BASE_DIR / "history.json"
USERS_FILE = BASE_DIR / "users.json"
LOGIN_HISTORY_FILE = BASE_DIR / "login_history.json"


def _configured_efficiency_loss() -> float | None:
    """Calculate loss from server-side physical readings when configured."""
    names = (
        "CLEAN_POWER_W",
        "DUSTY_POWER_W",
        "CLEAN_IRRADIANCE_W_M2",
        "DUSTY_IRRADIANCE_W_M2",
    )
    values = [os.environ.get(name) for name in names]
    if any(value in (None, "") for value in values):
        return None
    try:
        return calculate_efficiency_loss(*[float(value) for value in values])
    except (TypeError, ValueError):
        return None


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "The uploaded file is too large. Maximum size is 512 MB."}), 413

ALLOWED_IMAGES = {"png", "jpg", "jpeg", "webp"}
ALLOWED_VIDEOS = {"mp4", "avi", "mov", "mkv", "webm"}


def _allowed(filename: str, extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def _history_data() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        payload = json.loads(HISTORY_FILE.read_text())
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _users_data() -> dict[str, dict]:
    if not USERS_FILE.exists():
        return {}
    try:
        payload = json.loads(USERS_FILE.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _ensure_demo_user() -> dict[str, dict]:
    users = _users_data()
    if "admin" not in users:
        users["admin"] = {
            "display_name": "admin",
            "password_hash": generate_password_hash("admin123"),
        }
        _save_users(users)
    return users


def _save_users(users: dict[str, dict]) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2))


def _login_history_data() -> list[dict]:
    if not LOGIN_HISTORY_FILE.exists():
        return []
    try:
        payload = json.loads(LOGIN_HISTORY_FILE.read_text())
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _record_login(username: str, method: str) -> None:
    events = _login_history_data()
    events.append({
        "user": username,
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "method": method,
    })
    LOGIN_HISTORY_FILE.write_text(json.dumps(events, indent=2))


def _save_history(user: str, result: dict) -> None:
    history = _history_data()
    entry = {
        "user": user,
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "label": result.get("label"),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "dust_coverage": float(result.get("dust_coverage", 0.0) or 0.0),
        "efficiency_loss": result.get("efficiency_loss"),
        "next_cleaning": result.get("next_cleaning"),
        "action": result.get("action"),
    }
    history.append(entry)
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def _write_video_status(uid: str, payload: dict) -> None:
    """Replace a job status atomically so polling never reads half-written JSON."""
    status_path = VIDEO_RESULT_DIR / f"{uid}_status.json"
    temporary_path = VIDEO_RESULT_DIR / f"{uid}_{uuid.uuid4().hex}.tmp"
    temporary_path.write_text(json.dumps(payload, indent=2))
    try:
        for attempt in range(5):
            try:
                temporary_path.replace(status_path)
                return
            except PermissionError:
                if attempt == 4:
                    status_path.write_text(json.dumps(payload, indent=2))
                    return
                time.sleep(0.08)
    finally:
        temporary_path.unlink(missing_ok=True)


def _image_result(file_path: Path, uid: str) -> dict:
    image_bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Uploaded image could not be decoded.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    label, confidence, _probabilities = predict_rgb(model, image_rgb)
    overlay_rgb, heatmap_rgb, coverage = apply_dust_heatmap(image_bgr, model)
    coverage_source = "Grad-CAM estimate"
    if segmentation_model is not None:
        _mask, coverage = predict_dust_mask(segmentation_model, image_bgr)
        coverage_source = "U-Net pixel segmentation"

    overlay_name = f"overlay_{uid}.png"
    heatmap_name = f"heatmap_{uid}.png"
    cv2.imwrite(
        str(RESULT_DIR / overlay_name),
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(RESULT_DIR / heatmap_name),
        cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR),
    )

    analysis = analyze_dust(confidence, coverage, label)
    return {
        "label": label,
        "confidence": analysis.confidence,
        "dust_coverage": analysis.dust_coverage,
        "coverage_source": coverage_source,
        "dust_level": analysis.dust_level,
        "level_color": analysis.level_color,
        "efficiency_loss": _configured_efficiency_loss(),
        "action": analysis.maintenance_action,
        "priority": analysis.maintenance_priority,
        "next_cleaning": analysis.next_cleaning_date,
        "recommendations": analysis.recommendations,
        "original_img": f"/uploads/{file_path.name}",
        "overlay_img": f"/results/{overlay_name}",
        "heatmap_img": f"/results/{heatmap_name}",
    }


def _login_required(view):
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login_page"))
        return view(*args, **kwargs)
    wrapper.__name__ = view.__name__
    return wrapper


@app.get("/")
def index():
    if session.get("user"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login_page"))


@app.get("/login")
def login_page():
    return render_template("login.html")


@app.post("/login")
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    users = _ensure_demo_user()
    registered_user = users.get(username.lower())
    valid_registered_login = registered_user and check_password_hash(
        registered_user["password_hash"], password
    )
    if valid_registered_login:
        session["user"] = username
        _record_login(username, "password")
        return redirect(url_for("dashboard"))
    return render_template("login.html", error="Invalid username or password."), 401


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")
    if len(username) < 3:
        return render_template("register.html", error="Username must be at least 3 characters."), 400
    if len(password) < 8:
        return render_template("register.html", error="Password must be at least 8 characters."), 400
    if password != confirm_password:
        return render_template("register.html", error="Passwords do not match."), 400

    users = _users_data()
    username_key = username.lower()
    if username_key == "admin" or username_key in users:
        return render_template("register.html", error="That username is already registered."), 409

    users[username_key] = {
        "display_name": username,
        "password_hash": generate_password_hash(password),
    }
    _save_users(users)
    session["user"] = username
    _record_login(username, "account_created")
    return redirect(url_for("dashboard"))


@app.get("/auth/google")
def google_login():
    return render_template(
        "login.html",
        error="Google sign-in needs GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET configuration.",
    ), 501


@app.get("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login_page"))


@app.get("/dashboard")
@_login_required
def dashboard():
    return render_template("index.html", user=session.get("user"))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


@app.post("/efficiency_loss")
@_login_required
def efficiency_loss():
    payload = request.get_json(silent=True) or request.form
    try:
        value = calculate_efficiency_loss(
            payload.get("clean_power_w"),
            payload.get("dusty_power_w"),
            payload.get("clean_irradiance_w_m2"),
            payload.get("dusty_irradiance_w_m2"),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"efficiency_loss": value, "unit": "percent", "measured": True})


@app.get("/evaluate")
@_login_required
def evaluate():
    dataset_dir = None
    for candidate in (BASE_DIR / "dataset", BASE_DIR / "dataset" / "test", BASE_DIR / "dataset" / "train"):
        if candidate.exists() and next(candidate.iterdir(), None) is not None:
            dataset_dir = candidate
            break
    if dataset_dir is None:
        return jsonify({
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "dataset": "not-found",
        })

    metrics = evaluate_flat_dataset(model, dataset_dir, test_fraction=0.2, batch_size=8)
    return jsonify(metrics)


@app.post("/predict")
@_login_required
def predict():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "No image file uploaded."}), 400
    if not _allowed(uploaded.filename, ALLOWED_IMAGES):
        return jsonify({"error": "Unsupported image type."}), 400

    uid = uuid.uuid4().hex[:10]
    filename = secure_filename(f"{uid}_{uploaded.filename}")
    path = UPLOAD_DIR / filename
    uploaded.save(path)
    try:
        result = _image_result(path, uid)
        _save_history(session["user"], result)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422


def _run_video(input_path: Path, output_path: Path, uid: str, user: str, skip: int, heatmap: bool):
    try:
        def progress(percent: int, stats: dict) -> None:
            # The API remains usable without Flask-SocketIO.  Progress is
            # persisted for polling at /video_status/<uid>.
            _write_video_status(uid, {
                "status": "processing",
                "percent": percent,
                **stats,
            })

        summary = process_video(
            input_path=input_path,
            output_path=output_path,
            model=model,
            skip_frames=skip,
            show_heatmap=heatmap,
            progress_cb=progress,
            segmentation_model=segmentation_model,
        )
        overall_label = summary.get("overall_label", "Clean")
        maintenance_label = (
            "Dusty"
            if summary.get("avg_coverage", 0.0) >= 15.0
            else overall_label
        )
        video_analysis = analyze_dust(
            summary.get("avg_confidence", 0.0),
            summary.get("avg_coverage", 0.0),
            maintenance_label,
        )
        _save_history(user, {
            "label": overall_label,
            "confidence": video_analysis.confidence,
            "dust_coverage": video_analysis.dust_coverage,
            "efficiency_loss": video_analysis.efficiency_loss,
            "next_cleaning": video_analysis.next_cleaning_date,
            "action": video_analysis.maintenance_action,
        })
        summary.update({
            "label": overall_label,
            "maintenance_label": maintenance_label,
            "confidence": video_analysis.confidence,
            "dust_coverage": video_analysis.dust_coverage,
            "dust_level": video_analysis.dust_level,
            "efficiency_loss": video_analysis.efficiency_loss,
            "action": video_analysis.maintenance_action,
            "next_cleaning": video_analysis.next_cleaning_date,
            "recommendations": video_analysis.recommendations,
            "efficiency_loss": _configured_efficiency_loss(),
            "status": "complete",
            "video_url": f"/videos/{output_path.name}",
        })
        _write_video_status(uid, summary)
    except Exception as exc:
        _write_video_status(uid, {"status": "error", "error": str(exc)})


@app.post("/predict_video")
@_login_required
def predict_video():
    uploaded = request.files.get("video")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "No video file uploaded."}), 400
    if not _allowed(uploaded.filename, ALLOWED_VIDEOS):
        return jsonify({"error": "Unsupported video type."}), 400

    try:
        skip = max(0, int(request.form.get("skip_frames", "12")))
    except ValueError:
        return jsonify({"error": "skip_frames must be an integer."}), 400
    heatmap = request.form.get("heatmap", "true").lower() == "true"

    uid = uuid.uuid4().hex[:10]
    input_name = secure_filename(f"{uid}_{uploaded.filename}")
    input_path = VIDEO_UPLOAD_DIR / input_name
    output_path = VIDEO_RESULT_DIR / f"result_{uid}.mp4"
    uploaded.save(input_path)
    _write_video_status(uid, {"status": "queued", "percent": 0})

    thread = threading.Thread(
        target=_run_video,
        args=(input_path, output_path, uid, session["user"], skip, heatmap),
        daemon=True,
    )
    thread.start()
    return jsonify({"status": "processing", "video_id": uid}), 202


@app.get("/video_status/<uid>")
@_login_required
def video_status(uid: str):
    status_path = VIDEO_RESULT_DIR / f"{secure_filename(uid)}_status.json"
    if not status_path.exists():
        return jsonify({"error": "Video job not found."}), 404
    try:
        return jsonify(json.loads(status_path.read_text()))
    except (OSError, json.JSONDecodeError):
        return jsonify({"status": "processing", "percent": 0}), 200


@app.get("/history")
@_login_required
def history():
    user = session.get("user")
    items = [item for item in _history_data() if item.get("user") == user]
    return jsonify(items)


@app.get("/login_history")
@_login_required
def login_history():
    user = session.get("user")
    events = [item for item in _login_history_data() if item.get("user") == user]
    return jsonify({
        "count": len(events),
        "last_sign_in": events[-1] if events else None,
        "events": events,
    })


@app.get("/uploads/<path:filename>")
def uploads(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/results/<path:filename>")
def results(filename: str):
    return send_from_directory(RESULT_DIR, filename)


@app.get("/videos/<path:filename>")
def videos(filename: str):
    return send_from_directory(VIDEO_RESULT_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
