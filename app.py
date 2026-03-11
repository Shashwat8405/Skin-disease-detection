"""
Skin Disease Detection Server
Accepts: base64 image + sensor data (temperature, color)
Returns: disease prediction + skin health score (0-100)

Install dependencies:
  pip install flask flask-cors tensorflow pillow numpy
"""

import os
import base64
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# TensorFlow import with error handling
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print("[OK] TensorFlow loaded")
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not available — running in demo mode")

app = Flask(__name__)
CORS(app)

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH       = "skin_disease_model.h5"
CLASS_LABELS_PATH = "class_labels.txt"
IMG_SIZE         = 224

# Load class labels from file (same as your original server)
if os.path.exists(CLASS_LABELS_PATH):
    with open(CLASS_LABELS_PATH, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]
    print(f"[OK] Classes loaded from file: {CLASSES}")
else:
    CLASSES = ["acne", "eczema", "fungal", "normal"]
    print(f"[WARN] class_labels.txt not found — using defaults: {CLASSES}")

# Disease info: base_score, description, recommendations
# Add any new classes here if your class_labels.txt has more
DISEASE_INFO = {
    "normal": {
        "base_score": 90,
        "label": "Normal / Healthy",
        "recommendations": [
            "Maintain your current skincare routine",
            "Stay hydrated and use SPF daily",
            "Regular moisturizing recommended"
        ]
    },
    "acne": {
        "base_score": 55,
        "label": "Acne Detected",
        "recommendations": [
            "Use non-comedogenic skincare products",
            "Cleanse face twice daily with gentle cleanser",
            "Avoid touching face frequently",
            "Consider consulting a dermatologist"
        ]
    },
    "eczema": {
        "base_score": 45,
        "label": "Eczema Detected",
        "recommendations": [
            "Moisturize frequently with fragrance-free products",
            "Avoid known irritants and allergens",
            "Use mild, unscented soaps",
            "Consult a dermatologist for prescription treatment"
        ]
    },
    "fungal": {
        "base_score": 40,
        "label": "Fungal Infection Detected",
        "recommendations": [
            "Keep affected area clean and dry",
            "Avoid sharing personal items",
            "Use antifungal cream as directed",
            "See a doctor if symptoms persist"
        ]
    }
}

# ─── Load Model ───────────────────────────────────────────────────────────────
model = None
if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[OK] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
else:
    if TF_AVAILABLE:
        print(f"[WARN] Model file '{MODEL_PATH}' not found — using demo predictions")
    print("[INFO] Place your trained skin_model.h5 in the server/ directory")


# ─── Image Processing ─────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize image for MobileNetV2."""
    img = image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_disease(image: Image.Image):
    """Run model inference. Returns (class_name, confidence_percent, all_predictions)."""
    if model is not None:
        tensor = preprocess_image(image)
        preds  = model.predict(tensor, verbose=0)[0]
        idx    = int(np.argmax(preds))
        conf   = float(preds[idx]) * 100.0
        all_preds = {CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASSES))}
        return CLASSES[idx], conf, all_preds
    else:
        import random
        cls  = random.choice(CLASSES)
        conf = random.uniform(72, 94)
        all_preds = {c: round(random.uniform(1, 10), 2) for c in CLASSES}
        all_preds[cls] = round(conf, 2)
        return cls, conf, all_preds


def get_disease_info(disease: str) -> dict:
    """Get disease info, with fallback for unknown classes."""
    if disease in DISEASE_INFO:
        return DISEASE_INFO[disease]
    # Fallback for any class not in DISEASE_INFO
    return {
        "base_score": 50,
        "label": disease.replace("_", " ").title() + " Detected",
        "recommendations": [
            "Please consult a dermatologist for proper diagnosis",
            "Keep the affected area clean",
            "Avoid self-medication"
        ]
    }


# ─── Sensor Analysis ──────────────────────────────────────────────────────────
def analyze_temperature(temp: float) -> dict:
    """
    Normal skin surface temp: 30–37°C
    Inflamed skin: >37°C
    """
    if temp < 30:
        return {"status": "low", "label": f"{temp:.1f}°C — Below normal", "score_modifier": -5}
    elif temp <= 37:
        return {"status": "normal", "label": f"{temp:.1f}°C — Normal", "score_modifier": 0}
    elif temp <= 39:
        return {"status": "elevated", "label": f"{temp:.1f}°C — Slightly elevated (possible inflammation)", "score_modifier": -10}
    else:
        return {"status": "high", "label": f"{temp:.1f}°C — High (inflammation likely)", "score_modifier": -20}


def analyze_color(r: int, g: int, b: int, redness: float) -> dict:
    """
    Redness index analysis.
    redness = (R / (R+G+B)) * 100
    Normal skin: redness ~35–45%
    Inflamed/irritated: redness >50%
    """
    if redness < 30:
        return {"status": "pale", "label": "Pale / low pigmentation", "score_modifier": -5}
    elif redness <= 45:
        return {"status": "normal", "label": "Normal skin tone", "score_modifier": 0}
    elif redness <= 55:
        return {"status": "mild_redness", "label": "Mild redness / irritation", "score_modifier": -8}
    elif redness <= 65:
        return {"status": "moderate_redness", "label": "Moderate redness / inflammation", "score_modifier": -15}
    else:
        return {"status": "high_redness", "label": "High redness — significant inflammation", "score_modifier": -25}


def calculate_skin_score(base_score: float, confidence: float,
                          temp_analysis: dict, color_analysis: dict) -> int:
    """
    Final skin score (0–100):
      base_score       = disease base (normal=90, acne=55, etc.)
      confidence bonus = higher AI confidence → closer to base score
      temp modifier    = sensor-based adjustment
      color modifier   = sensor-based adjustment
    """
    # Confidence scaling: at 100% confidence, full base score. At 50%, reduce by up to 15pts
    confidence_factor = (confidence - 50) / 50.0  # 0.0 to 1.0
    confidence_factor = max(0.0, min(1.0, confidence_factor))
    score = base_score * (0.85 + 0.15 * confidence_factor)

    # Apply sensor modifiers
    score += temp_analysis["score_modifier"]
    score += color_analysis["score_modifier"]

    return int(max(0, min(100, round(score))))


def score_to_grade(score: int) -> dict:
    if score >= 85:
        return {"grade": "A", "label": "Excellent", "color": "#22c55e"}
    elif score >= 70:
        return {"grade": "B", "label": "Good", "color": "#84cc16"}
    elif score >= 55:
        return {"grade": "C", "label": "Fair", "color": "#f59e0b"}
    elif score >= 40:
        return {"grade": "D", "label": "Poor", "color": "#f97316"}
    else:
        return {"grade": "F", "label": "Critical", "color": "#ef4444"}


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Serve the web app — open http://YOUR_PC_IP:5000 in Chrome on your phone"""
    return send_from_directory("templates", "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "mode": "live" if model is not None else "demo"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Expected JSON body:
    {
      "image": "<base64 encoded JPEG/PNG>",
      "temperature": 34.5,
      "red": 180,
      "green": 140,
      "blue": 110,
      "redness": 42.5
    }
    """
    try:
        data = request.get_json(force=True)

        # ── 1. Decode Image ──────────────────────────────────────────────────
        if "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        img_b64 = data["image"]
        # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,...")
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        img_bytes = base64.b64decode(img_b64)
        image     = Image.open(io.BytesIO(img_bytes))

        # ── 2. Extract Sensor Data ───────────────────────────────────────────
        temperature = float(data.get("temperature", 33.0))
        red         = int(data.get("red",         180))
        green       = int(data.get("green",       140))
        blue        = int(data.get("blue",        110))
        redness     = float(data.get("redness",   42.0))

        # ── 3. AI Prediction ─────────────────────────────────────────────────
        disease, confidence, all_preds = predict_disease(image)

        # ── 4. Sensor Analysis ───────────────────────────────────────────────
        temp_analysis  = analyze_temperature(temperature)
        color_analysis = analyze_color(red, green, blue, redness)

        # ── 5. Calculate Score ───────────────────────────────────────────────
        info       = get_disease_info(disease)
        base_score = info["base_score"]
        skin_score = calculate_skin_score(base_score, confidence,
                                          temp_analysis, color_analysis)
        grade_info = score_to_grade(skin_score)

        # ── 6. Build Response ────────────────────────────────────────────────
        response = {
            "disease": {
                "name":            disease,
                "label":           info["label"],
                "confidence":      round(confidence, 1),
                "all_predictions": all_preds,
                "recommendations": info["recommendations"]
            },
            "skin_score": {
                "score":    skin_score,
                "grade":    grade_info["grade"],
                "label":    grade_info["label"],
                "color":    grade_info["color"],
                "max":      100
            },
            "sensors": {
                "temperature": {
                    "value":  temperature,
                    "unit":   "°C",
                    "status": temp_analysis["status"],
                    "label":  temp_analysis["label"]
                },
                "color": {
                    "r":      red,
                    "g":      green,
                    "b":      blue,
                    "redness_index": round(redness, 1),
                    "status": color_analysis["status"],
                    "label":  color_analysis["label"]
                }
            },
            "score_breakdown": {
                "base_score":        base_score,
                "confidence_boost":  round(skin_score - base_score
                                           - temp_analysis["score_modifier"]
                                           - color_analysis["score_modifier"], 1),
                "temp_modifier":     temp_analysis["score_modifier"],
                "color_modifier":    color_analysis["score_modifier"],
                "final_score":       skin_score
            }
        }

        return jsonify(response), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Skin Disease Detection Server v2.0 ===")
    print("Listening on http://0.0.0.0:5000")
    print("Make sure your phone and ESP32 are on the same WiFi network\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
