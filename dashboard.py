"""
dashboard.py — SolarScan AI desktop dashboard
Requires: model.py in the same folder, model.pth trained weights

Usage:
    python dashboard.py
"""

import datetime
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision import transforms

from model import load_model, CLASSES, device   # single source of truth

# --------------------------------
# Transform — matches val_transform in train_v2.py
# --------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

OUTPUT_DIR = "outputs/frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------
# Load model once at startup
# --------------------------------
try:
    model = load_model()
except FileNotFoundError as e:
    model = None
    _MODEL_ERROR = str(e)

# --------------------------------
# Inference helpers
# --------------------------------
@torch.no_grad()
def predict(bgr_frame):
    """Run inference on a BGR frame. Returns result dict."""
    rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)

    probs      = F.softmax(model(tensor), dim=1)
    conf, pred = torch.max(probs, 1)
    label      = CLASSES[pred.item()]
    confidence = conf.item() * 100

    if label == "Clean":
        efficiency = round(95 + (confidence / 100) * 5, 2)
    else:
        efficiency = max(0.0, round(60 - (confidence / 100) * 20, 2))

    if label == "Dusty" and confidence > 70:
        maintenance = "IMMEDIATE CLEANING REQUIRED"
    elif label == "Dusty":
        maintenance = "Cleaning recommended soon"
    else:
        maintenance = "No maintenance required"

    today         = datetime.date.today()
    next_cleaning = today + datetime.timedelta(days=2 if label == "Dusty" else 15)

    return {
        "label":         label,
        "confidence":    round(confidence, 2),
        "efficiency":    efficiency,
        "maintenance":   maintenance,
        "next_cleaning": str(next_cleaning),
    }


def predict_video_frames(path, skip=10, max_frames=300):
    """Analyse multiple frames and return aggregate result + per-frame list."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    results, frame_idx, sampled = [], 0, 0
    first_frame_rgb = None

    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            if first_frame_rgb is None:
                first_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results.append(predict(frame))
            sampled += 1
        frame_idx += 1

    cap.release()

    if not results:
        raise RuntimeError("No frames could be extracted from the video.")

    dusty  = sum(1 for r in results if r["label"] == "Dusty")
    ratio  = dusty / len(results)
    final  = "Dusty" if ratio > 0.5 else "Clean"
    avg_c  = round(sum(r["confidence"] for r in results) / len(results), 2)
    avg_e  = round(sum(r["efficiency"] for r in results) / len(results), 2)

    today         = datetime.date.today()
    next_cleaning = today + datetime.timedelta(days=2 if final == "Dusty" else 15)

    if final == "Dusty" and avg_c > 70:
        maintenance = "IMMEDIATE CLEANING REQUIRED"
    elif final == "Dusty":
        maintenance = "Cleaning recommended soon"
    else:
        maintenance = "No maintenance required"

    summary = {
        "label":          final,
        "confidence":     avg_c,
        "efficiency":     avg_e,
        "maintenance":    maintenance,
        "next_cleaning":  str(next_cleaning),
        "frames_total":   len(results),
        "dusty_frames":   dusty,
        "clean_frames":   len(results) - dusty,
        "dust_ratio":     round(ratio * 100, 2),
    }
    return summary, first_frame_rgb

# --------------------------------
# Colour theme
# --------------------------------
BG        = "#0f1117"
CARD      = "#1a1d27"
ACCENT    = "#00d4aa"
ACCENT2   = "#4f8ef7"
TEXT      = "#e8eaf0"
TEXT_MUTE = "#6b7280"
RED       = "#ef4444"
GREEN     = "#22c55e"
AMBER     = "#f59e0b"
FONT_H1   = ("Segoe UI", 22, "bold")
FONT_H2   = ("Segoe UI", 14, "bold")
FONT_BODY = ("Segoe UI", 11)
FONT_MONO = ("Consolas", 11)

# --------------------------------
# Reusable widgets
# --------------------------------
def card(parent, **kwargs):
    return tk.Frame(parent, bg=CARD, bd=0, relief="flat", **kwargs)

def label(parent, text="", font=FONT_BODY, fg=TEXT, bg=CARD, anchor="w", **kwargs):
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, anchor=anchor, **kwargs)

def btn(parent, text, command, color=ACCENT, fg=BG, width=18):
    b = tk.Button(
        parent, text=text, command=command,
        bg=color, fg=fg, activebackground=color, activeforeground=fg,
        font=("Segoe UI", 11, "bold"), relief="flat", bd=0,
        cursor="hand2", width=width, pady=8,
    )
    b.bind("<Enter>", lambda e: b.config(bg=_lighten(color)))
    b.bind("<Leave>", lambda e: b.config(bg=color))
    return b

def _lighten(hex_col):
    r, g, b = int(hex_col[1:3],16), int(hex_col[3:5],16), int(hex_col[5:7],16)
    r, g, b = min(255,r+30), min(255,g+30), min(255,b+30)
    return f"#{r:02x}{g:02x}{b:02x}"

def separator(parent, pady=6):
    tk.Frame(parent, bg=TEXT_MUTE, height=1).pack(fill="x", padx=20, pady=pady)

# --------------------------------
# Result window
# --------------------------------
def show_result(root, title, img_rgb, result, extra_lines=None):
    win = tk.Toplevel(root)
    win.title(f"SolarScan — {title}")
    win.configure(bg=BG)
    win.resizable(False, False)

    # Image panel
    img_pil = Image.fromarray(img_rgb).resize((480, 320))
    img_tk  = ImageTk.PhotoImage(img_pil)
    img_lbl = tk.Label(win, image=img_tk, bg=BG, bd=0)
    img_lbl.image = img_tk
    img_lbl.pack(padx=20, pady=(20, 8))

    # Result badge
    is_dusty = result["label"] == "Dusty"
    badge_bg = RED if is_dusty else GREEN
    badge_tx = "⚠  DUSTY" if is_dusty else "✔  CLEAN"
    tk.Label(win, text=badge_tx, bg=badge_bg, fg="white",
             font=("Segoe UI", 13, "bold"), pady=6, padx=20).pack(pady=(4, 12))

    # Metrics card
    c = card(win)
    c.pack(padx=20, pady=4, fill="x")

    rows = [
        ("Prediction",    result["label"]),
        ("Confidence",    f"{result['confidence']:.2f}%"),
        ("Panel efficiency", f"{result['efficiency']}%"),
        ("Maintenance",   result["maintenance"]),
        ("Next cleaning", result["next_cleaning"]),
    ]
    if extra_lines:
        rows += extra_lines

    for key, val in rows:
        row = tk.Frame(c, bg=CARD)
        row.pack(fill="x", padx=16, pady=3)
        label(row, text=key, fg=TEXT_MUTE, width=22).pack(side="left")
        label(row, text=val, fg=TEXT,      font=FONT_MONO).pack(side="left")

    separator(win, pady=8)
    btn(win, "Close", win.destroy, color=TEXT_MUTE, fg=BG, width=12).pack(pady=(0,16))


# ================================
# Main App
# ================================
class SolarScanApp:
    def __init__(self, root):
        self.root = root
        root.title("SolarScan AI — Dashboard")
        root.configure(bg=BG)
        root.geometry("640x480")
        root.resizable(False, False)

        self._build_ui()

        if model is None:
            messagebox.showerror("Model Error", _MODEL_ERROR)

    # ---- UI layout ----
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=30, pady=(28, 0))
        tk.Label(hdr, text="☀  SolarScan AI", font=FONT_H1,
                 fg=ACCENT, bg=BG).pack(side="left")
        status_col = GREEN if model else RED
        status_txt = "Model ready" if model else "Model missing"
        tk.Label(hdr, text=f"● {status_txt}", font=FONT_BODY,
                 fg=status_col, bg=BG).pack(side="right", pady=6)

        separator(self.root, pady=12)

        # Action cards row
        row = tk.Frame(self.root, bg=BG)
        row.pack(padx=30, fill="x")

        self._action_card(row, "🖼  Image", "Analyse a single image\nfor dust detection",
                          self._predict_image, ACCENT)
        tk.Frame(row, bg=BG, width=20).pack(side="left")
        self._action_card(row, "🎬  Video", "Analyse multiple frames\nfrom a video file",
                          self._predict_video, ACCENT2)

        separator(self.root, pady=18)

        # Status bar
        self.status_var = tk.StringVar(value="Ready — select an input above.")
        tk.Label(self.root, textvariable=self.status_var,
                 font=FONT_BODY, fg=TEXT_MUTE, bg=BG).pack(pady=4)

        # Progress bar (hidden until needed)
        self.progress = ttk.Progressbar(self.root, mode="indeterminate", length=400)

    def _action_card(self, parent, title, subtitle, command, accent):
        c = card(parent)
        c.pack(side="left", fill="both", expand=True, ipady=16)

        tk.Label(c, text=title, font=FONT_H2, fg=accent, bg=CARD).pack(pady=(16,4), padx=20)
        tk.Label(c, text=subtitle, font=FONT_BODY, fg=TEXT_MUTE,
                 bg=CARD, justify="center").pack(padx=20)
        btn(c, "Select file", command, color=accent).pack(pady=16)

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _guard(self):
        if model is None:
            messagebox.showerror("Model Error", "model.pth not loaded.")
            return False
        return True

    # ---- Image prediction ----
    def _predict_image(self):
        if not self._guard():
            return
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Cannot read image:\n{path}")
            return

        self._set_status(f"Analysing {os.path.basename(path)} …")
        result  = predict(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._set_status("Done.")
        show_result(self.root, "Image Result", img_rgb, result)

    # ---- Video prediction (threaded so UI doesn't freeze) ----
    def _predict_video(self):
        if not self._guard():
            return
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not path:
            return

        self._set_status(f"Analysing {os.path.basename(path)} …")
        self.progress.pack(pady=8)
        self.progress.start(12)

        def _run():
            try:
                summary, first_frame = predict_video_frames(path, skip=10, max_frames=300)
                self.root.after(0, lambda: self._on_video_done(summary, first_frame))
            except Exception as exc:
                self.root.after(0, lambda: self._on_video_error(str(exc)))

        threading.Thread(target=_run, daemon=True).start()

    def _on_video_done(self, summary, first_frame):
        self.progress.stop()
        self.progress.pack_forget()
        self._set_status("Done.")

        extra = [
            ("Frames analysed",  str(summary["frames_total"])),
            ("Dusty frames",     f"{summary['dusty_frames']}  ({summary['dust_ratio']}%)"),
            ("Clean frames",     str(summary["clean_frames"])),
        ]
        show_result(self.root, "Video Result", first_frame, summary, extra_lines=extra)

    def _on_video_error(self, msg):
        self.progress.stop()
        self.progress.pack_forget()
        self._set_status("Error during video analysis.")
        messagebox.showerror("Video Error", msg)


# --------------------------------
# Run
# --------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app  = SolarScanApp(root)
    root.mainloop()