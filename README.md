# 🩺 Skin Analysis — IoT + AI Disease Detection & Health Scoring

> **Real-time skin disease detection using ESP32 BLE sensors + MobileNetV2 deep learning, accessible entirely from a smartphone browser — no app install required.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.10-orange?logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-2.3-lightgrey?logo=flask&logoColor=black"/>
  <img src="https://img.shields.io/badge/ESP32-BLE%204.2-red?logo=espressif&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accuracy-88%25-brightgreen"/>
  <img src="https://img.shields.io/badge/Hardware%20Cost-₹1%2C100-yellow"/>
</p>

---

## 📸 Screenshots

| Step 1 — Connect Sensor | Step 2 — Capture Photo | Step 3 — Health Score | Step 4 — Results |
|:-:|:-:|:-:|:-:|
| ![Sensor](screenshots/step1_sensor.png) | ![Photo](screenshots/step2_photo.png) | ![Score](screenshots/step3_score.png) | ![Results](screenshots/step4_results.png) |

> *Detected: Fungal Infection (51.6% confidence) · Health Score: 34/100 · Grade: F (Critical)*

---

## 🧠 What It Does

A **4-step browser-based workflow** — no app, no cloud, no sign-up:

1. **Connect** — Chrome pairs with the ESP32 over Web Bluetooth
2. **Capture** — Take a skin photo with your phone camera
3. **Analyze** — MobileNetV2 classifies the image; sensor data adjusts the score
4. **Results** — View disease detection, confidence, health score (0–100), and recommendations

The system detects **4 skin conditions**: Acne · Eczema · Fungal Infection · Normal Skin

---

## 🏗️ System Architecture

```
📱 Smartphone (Chrome for Android)
    │
    ├── [Web Bluetooth] ←──────────────────── ESP32 DevKit v1
    │                                              ├── MLX90614  (skin temperature, ±0.5°C)
    │                                              └── TCS34725  (RGB color / redness index)
    │
    └── [HTTP POST / Wi-Fi] ──────────────── Flask Server (local PC)
                                                   └── MobileNetV2 (fine-tuned, 4 classes)
                                                         └── Health Scoring Algorithm
                                                               └── JSON response → UI
```

---

## ✨ Key Features

- **No app install** — runs entirely in Chrome browser via Web Bluetooth API
- **Dual-sensor fusion** — infrared temperature + RGB redness index improve diagnostic accuracy
- **Composite health score** — 0–100 score combining CNN confidence + physiological sensor readings
- **Sensor-free mode** — works without hardware using default physiological values
- **Real-time inference** — 0.89 s mean end-to-end latency (image submission → results)
- **Responsive UI** — works on 360 px mobile screens; clean 4-step guided flow

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Overall Accuracy (50 real samples) | **88.0%** |
| Normal Skin Accuracy | **100%** (0 false positives) |
| Core Inference Latency | **0.89 ± 0.14 s** |
| Model Parameters | **3.4M** (MobileNetV2) |
| Hardware Cost | **₹1,100 (~$13 USD)** |

**Per-class F1 scores (test set, n = 800):**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Acne | 0.84 | 0.87 | 0.85 |
| Eczema | 0.88 | 0.85 | 0.86 |
| Fungal | 0.82 | 0.88 | 0.85 |
| Normal | 0.93 | 0.90 | **0.91** |

---

## 🔬 Health Scoring Algorithm

```
Score = clamp(B + C + T + R, 0, 100)
```

| Component | Description | Range |
|-----------|-------------|-------|
| **B** — Base | Disease severity (Normal=90, Acne=55, Eczema=45, Fungal=40) | 40–90 |
| **C** — Confidence | CNN confidence modifier (+10 / +5 / 0) | 0–+10 |
| **T** — Temperature | Skin temp deviation (0 / −8 / −18) | −18–0 |
| **R** — Redness | RGB redness index (0 / −7 / −15 / −25) | −25–0 |

**Grades:** A (85–100) · B (70–84) · C (55–69) · D (40–54) · F (0–39)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Deep Learning** | TensorFlow 2.10, Keras, MobileNetV2 (transfer learning) |
| **Backend** | Python 3.10, Flask 2.3, Flask-CORS, Pillow, NumPy |
| **Firmware** | ESP32 Arduino Core, ESP32 BLE Arduino, Adafruit MLX90614 & TCS34725 |
| **Frontend** | HTML5, CSS3, JavaScript ES6, Web Bluetooth API |
| **Model Format** | Keras HDF5 (14 MB) |

---

## ⚙️ Hardware

| Component | Role | Cost |
|-----------|------|------|
| ESP32 DevKit v1 | BLE GATT server + I²C master | ₹300 |
| MLX90614 IR Sensor | Non-contact skin temperature (±0.5°C) | ₹450 |
| TCS34725 Color Sensor | RGB redness index (16-bit RGBC) | ₹350 |
| **Total** | | **₹1,100** |

**Wiring:** Both sensors share I²C (SDA: GPIO21, SCL: GPIO22) with 4.7 kΩ pull-ups to 3.3 V.

---

## 🚀 Quick Start

### 1. Clone & install dependencies
```bash
git clone https://github.com/yourusername/skin-analysis
cd skin-analysis
pip install flask flask-cors tensorflow pillow numpy
```

### 2. Place your model
```
skin_analyzer/
├── app.py
├── skin_disease_model.h5    ← your trained model
├── class_labels.txt         ← acne, eczema, fungal, normal
└── templates/
      └── index.html
```

### 3. Run the server
```bash
cd skin_analyzer
python app.py
# Server starts at http://0.0.0.0:5000
```

### 4. Allow firewall (Windows)
```powershell
# Run as Administrator
netsh advfirewall firewall add rule name="Flask 5000" dir=in action=allow protocol=TCP localport=5000
```

### 5. Open on your phone
Open **Chrome for Android** and navigate to `http://<your-pc-ip>:5000`
*(Both devices must be on the same Wi-Fi network)*

### 6. Flash ESP32 (optional)
Open `esp32/skin_sensor_ble/skin_sensor_ble.ino` in Arduino IDE and flash to your ESP32.

---

## 📁 Project Structure

```
skin-analysis/
├── skin_analyzer/
│   ├── app.py                  # Flask REST server + inference logic
│   ├── skin_disease_model.h5   # Trained MobileNetV2 model
│   ├── class_labels.txt        # Class names
│   └── templates/
│       └── index.html          # Single-page browser app (Web Bluetooth + UI)
├── esp32/
│   └── skin_sensor_ble/
│       └── skin_sensor_ble.ino # ESP32 BLE GATT firmware
├── screenshots/
│   ├── step1_sensor.png
│   ├── step2_photo.png
│   ├── step3_score.png
│   └── step4_results.png
└── README.md
```

---

## 🔌 API Reference

### `POST /analyze`

**Request body (JSON):**
```json
{
  "image": "<base64-encoded JPEG>",
  "temperature": 35.2,
  "red": 180, "green": 140, "blue": 110,
  "redness": 42.1
}
```

**Response:**
```json
{
  "disease": "fungal",
  "confidence": 0.516,
  "score": 34,
  "grade": "F",
  "all_predictions": { "fungal": 0.516, "acne": 0.347, "normal": 0.134, "eczema": 0.004 },
  "score_breakdown": { "base": 40, "confidence": -6, "temperature": 0, "redness": 0 },
  "recommendations": ["Keep affected area clean and dry", "Use antifungal cream as directed", ...]
}
```

---

## 📱 Browser Compatibility

| Browser | BLE Support | Recommended |
|---------|------------|-------------|
| Chrome for Android | ✅ | ✅ Yes |
| Chrome Desktop | ✅ | ✅ Yes |
| Firefox | ❌ | ✗ No |
| Safari / iOS | ❌ | ✗ No |

---

## 🔭 Future Work

- [ ] TensorFlow Lite on-device inference (no Wi-Fi needed)
- [ ] Expand to 10 skin disease categories
- [ ] Cloud-hosted HTTPS backend with patient history
- [ ] Custom PCB wearable probe (ESP32 + sensors in 55 mm patch)
- [ ] iOS support via React Native BLE

---

## 📄 Publication

This project was submitted as a B.Tech final year project (Electronics & Telecommunication Engineering, KIIT University) and as an IEEE conference paper:

> *"IoT-Integrated Skin Disease Detection and Health Scoring Using MobileNetV2 Deep Learning"*
> 2025 IEEE International Conference on Electronics, Communication and Signal Processing (ICECSP-2025)

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not a medical device and should not be used for clinical diagnosis. Always consult a qualified dermatologist for medical advice.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
