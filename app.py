
import os, io, base64, sqlite3, re
from pathlib import Path
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "brain_tumor_secret_key_2024"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DB_PATH     = BASE_DIR / "users.db"
MODEL_PATH  = BASE_DIR / "outputs" / "brain_tumor_densenet121_full.pth"
UPLOAD_DIR  = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff"}
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname  TEXT    NOT NULL,
                email     TEXT    UNIQUE NOT NULL,
                password  TEXT    NOT NULL,
                phone     TEXT,
                gender    TEXT,
                dob       TEXT,
                created   TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                filename   TEXT,
                result     TEXT,
                confidence REAL,
                created    TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        conn.commit()

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = None

def load_model():
    global model
    if not MODEL_PATH.exists():
        print(f"[WARNING] Model not found at {MODEL_PATH}")
        return
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    global CLASS_NAMES
    CLASS_NAMES = checkpoint.get("class_names", CLASS_NAMES)

    m = models.densenet121(weights=None)
    in_feat = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_feat, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASS_NAMES)),
    )
    m.load_state_dict(checkpoint["model_state_dict"])
    m.to(device)
    m.eval()
    model = m
    print(f"[Model] Loaded from {MODEL_PATH} on {device}")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
def run_gradcam(input_tensor, pred_idx):
    """Returns base64-encoded overlay PNG."""
    gradients, activations = [], []

    fwd = model.features.denseblock4.register_forward_hook(
        lambda m, i, o: activations.append(o))
    bwd = model.features.denseblock4.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0]))

    out = model(input_tensor)
    model.zero_grad()
    out[0, pred_idx].backward()

    fwd.remove(); bwd.remove()

    grads = gradients[0].detach().cpu().numpy()[0]   # [C, H, W]
    acts  = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()

    heatmap  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Reconstruct original from tensor for overlay
    orig_np  = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mean_arr = np.array([0.485, 0.456, 0.406])
    std_arr  = np.array([0.229, 0.224, 0.225])
    orig_np  = np.clip(std_arr * orig_np + mean_arr, 0, 1)
    orig_u8  = (orig_np * 255).astype(np.uint8)

    overlay = cv2.addWeighted(orig_u8, 0.6, heatmap, 0.4, 0)

    # Encode to base64
    def to_b64(arr):
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return {
        "original": to_b64(orig_u8),
        "heatmap":  to_b64(heatmap),
        "overlay":  to_b64(overlay),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
TUMOR_INFO = {
    "glioma": {
        "label": "Glioma",
        "severity": "High",
        "badge": "danger",
        "description": (
            "Glioma is a type of tumor that starts in the glial cells of the brain "
            "or spine. It is one of the most common types of primary brain tumors and "
            "can be aggressive depending on its grade."
        ),
        "symptoms": [
            "Persistent headaches (worse in the morning)",
            "Seizures or convulsions",
            "Nausea and vomiting",
            "Memory loss or cognitive changes",
            "Vision or speech problems",
            "Weakness or numbness in limbs",
        ],
        "causes": [
            "Genetic mutations in glial cells",
            "Exposure to ionizing radiation",
            "Rare inherited conditions (Li-Fraumeni syndrome)",
            "Age (more common in adults 45–65)",
        ],
        "treatment": [
            "Surgical resection (tumor removal)",
            "Radiation therapy (targeted radiotherapy)",
            "Chemotherapy (Temozolomide)",
            "Targeted therapy & immunotherapy",
            "Corticosteroids (to reduce brain swelling)",
        ],
        "prevention": [
            "Minimize exposure to ionizing radiation",
            "Avoid unnecessary head CT scans",
            "Maintain a healthy lifestyle",
            "Regular neurological check-ups if at genetic risk",
            "Early reporting of persistent headaches",
        ],
        "prognosis": "Survival varies by grade. Low-grade gliomas may survive 5–10+ years; high-grade glioblastoma (GBM) averages 12–18 months with treatment.",
        "icon": "🧠",
    },
    "meningioma": {
        "label": "Meningioma",
        "severity": "Moderate",
        "badge": "warning",
        "description": (
            "Meningioma is a tumor that forms on the membranes (meninges) that "
            "cover the brain and spinal cord. Most meningiomas are benign and "
            "slow-growing, but some can be atypical or malignant."
        ),
        "symptoms": [
            "Gradual vision changes or double vision",
            "Hearing loss or ringing in ears",
            "Memory difficulties",
            "Headaches that worsen over time",
            "Weakness in arms or legs",
            "Personality or behavioral changes",
        ],
        "causes": [
            "Hormonal factors (more common in women)",
            "Previous radiation to the head",
            "Genetic disorders (Neurofibromatosis type 2)",
            "Obesity (possible risk factor)",
        ],
        "treatment": [
            "Active surveillance (watch and wait) for small tumors",
            "Surgical removal (craniotomy)",
            "Stereotactic radiosurgery (Gamma Knife)",
            "Radiation therapy for inoperable tumors",
            "Hormone therapy (experimental)",
        ],
        "prevention": [
            "Avoid unnecessary radiation exposure to the head",
            "Maintain healthy body weight",
            "Regular MRI monitoring if at risk",
            "Discuss hormone therapy risks with your doctor",
            "Genetic counseling for NF2 patients",
        ],
        "prognosis": "Most benign meningiomas have excellent outcomes after surgery. Recurrence is possible; 5-year survival rate exceeds 90% for benign cases.",
        "icon": "🛡️",
    },
    "pituitary": {
        "label": "Pituitary Tumor",
        "severity": "Moderate",
        "badge": "warning",
        "description": (
            "Pituitary tumors (adenomas) are abnormal growths that develop in the "
            "pituitary gland. They are almost always benign and affect hormone "
            "regulation in the body, causing a wide range of systemic symptoms."
        ),
        "symptoms": [
            "Vision problems (especially peripheral vision loss)",
            "Unexplained weight gain or loss",
            "Irregular menstrual cycles (in women)",
            "Erectile dysfunction (in men)",
            "Excessive growth of hands, feet, face (acromegaly)",
            "Fatigue and weakness",
            "Increased thirst and urination",
        ],
        "causes": [
            "Genetic mutations (MEN1 gene)",
            "Sporadic cell mutations in the pituitary gland",
            "Family history of pituitary disorders",
        ],
        "treatment": [
            "Medications (Dopamine agonists for prolactinomas)",
            "Transsphenoidal surgery (through the nose)",
            "Radiation therapy",
            "Hormone replacement therapy",
            "Regular MRI monitoring",
        ],
        "prevention": [
            "Regular hormone level checks if symptomatic",
            "Genetic screening for MEN1 families",
            "Routine eye exams to detect vision changes early",
            "Maintain hormonal health through regular endocrinology visits",
        ],
        "prognosis": "Prognosis is generally excellent. Most pituitary adenomas are benign and highly treatable. Surgery is curative in most cases.",
        "icon": "⚗️",
    },
    "notumor": {
        "label": "No Tumor Detected",
        "severity": "None",
        "badge": "success",
        "description": (
            "No brain tumor has been detected in the provided MRI scan. "
            "The scan appears normal. Continue maintaining good brain health "
            "through regular check-ups and a healthy lifestyle."
        ),
        "symptoms": [],
        "causes": [],
        "treatment": [
            "No treatment required",
            "Maintain regular neurological check-ups",
            "Report any new symptoms immediately",
        ],
        "prevention": [
            "Eat a balanced, brain-healthy diet (omega-3, antioxidants)",
            "Exercise regularly — improves blood flow to the brain",
            "Avoid smoking and limit alcohol consumption",
            "Protect your head from injuries (wear helmets)",
            "Manage stress through meditation and sleep",
            "Regular medical check-ups and MRI if at risk",
        ],
        "prognosis": "No tumor detected. Maintain regular health screenings and report any neurological symptoms promptly.",
        "icon": "✅",
    },
}

def predict_image(pil_img):
    if model is None:
        return None
    inp = TRANSFORM(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(inp)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    cls_name   = CLASS_NAMES[pred.item()]
    confidence = round(conf.item() * 100, 2)
    all_probs  = {CLASS_NAMES[i]: round(probs[0][i].item() * 100, 2)
                  for i in range(len(CLASS_NAMES))}

    gradcam_imgs = None
    if cls_name != "notumor":
        inp_grad    = TRANSFORM(pil_img).unsqueeze(0).to(device)
        inp_grad.requires_grad_(False)
        gradcam_imgs = run_gradcam(inp_grad, pred.item())

    return {
        "class":      cls_name,
        "confidence": confidence,
        "all_probs":  all_probs,
        "info":       TUMOR_INFO[cls_name],
        "gradcam":    gradcam_imgs,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  AUTH DECORATOR
# ─────────────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — AUTH
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
        if user and check_password_hash(user["password"], password):
            session["user_id"]  = user["id"]
            session["fullname"] = user["fullname"]
            session["email"]    = user["email"]
            flash(f"Welcome back, {user['fullname']}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")
        phone    = request.form.get("phone", "").strip()
        gender   = request.form.get("gender", "")
        dob      = request.form.get("dob", "")

        if not fullname or not email or not password:
            flash("All required fields must be filled.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        else:
            hashed = generate_password_hash(password)
            try:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO users (fullname,email,password,phone,gender,dob)"
                        " VALUES (?,?,?,?,?,?)",
                        (fullname, email, hashed, phone, gender, dob)
                    )
                    conn.commit()
                flash("Registration successful! Please login.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Email already registered.", "danger")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    with get_db() as conn:
        history = conn.execute(
            "SELECT * FROM predictions WHERE user_id=? ORDER BY created DESC LIMIT 5",
            (session["user_id"],)
        ).fetchall()
        total_preds = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE user_id=?",
            (session["user_id"],)
        ).fetchone()[0]
    return render_template("dashboard.html",
                           history=history,
                           total_preds=total_preds)

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        if "mri_image" not in request.files:
            flash("No file uploaded.", "danger")
            return redirect(request.url)
        file = request.files["mri_image"]
        if file.filename == "":
            flash("No file selected.", "danger")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Invalid file type. Upload PNG/JPG/JPEG/BMP/TIFF.", "danger")
            return redirect(request.url)
        if model is None:
            flash("Model not loaded. Place brain_tumor_densenet121_full.pth in outputs/.", "danger")
            return redirect(request.url)

        filename  = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(save_path)

        pil_img = Image.open(save_path).convert("RGB")
        result  = predict_image(pil_img)

        # Save prediction to DB
        with get_db() as conn:
            conn.execute(
                "INSERT INTO predictions (user_id,filename,result,confidence) VALUES (?,?,?,?)",
                (session["user_id"], filename,
                 result["class"], result["confidence"])
            )
            conn.commit()

        return render_template("result.html",
                               result=result,
                               image_url=url_for("static",
                                                 filename=f"uploads/{filename}"))
    return render_template("predict.html")

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES — PROFILE
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    with get_db() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE id=?", (session["user_id"],)
        ).fetchone()
        pred_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE user_id=?",
            (session["user_id"],)
        ).fetchone()[0]
        history = conn.execute(
            "SELECT * FROM predictions WHERE user_id=? ORDER BY created DESC",
            (session["user_id"],)
        ).fetchall()

    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        phone    = request.form.get("phone", "").strip()
        gender   = request.form.get("gender", "")
        dob      = request.form.get("dob", "")
        with get_db() as conn:
            conn.execute(
                "UPDATE users SET fullname=?,phone=?,gender=?,dob=? WHERE id=?",
                (fullname, phone, gender, dob, session["user_id"])
            )
            conn.commit()
        session["fullname"] = fullname
        flash("Profile updated successfully.", "success")
        return redirect(url_for("profile"))

    return render_template("profile.html",
                           user=user,
                           pred_count=pred_count,
                           history=history)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
