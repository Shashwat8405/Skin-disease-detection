# 🩺 Skin Analysis — IoT + AI Disease Detection & Health Scoring

> **Real-time skin disease detection using ESP32 BLE sensors + MobileNetV2 deep learning, accessible entirely from a smartphone browser — no app install required.**

<p align="center">
<img width="1919" height="975" alt="Screenshot 2026-03-12 002633" src="https://github.com/user-attachments/assets/262b5d29-b288-45a9-b4b1-13fe98079fe4" />
<img width="1916" height="979" alt="Screenshot 2026-03-12 002655" src="https://github.com/user-attachments/assets/30c712b0-563d-4ef7-a676-69371626adfa" />
<img width="1919" height="979" alt="Screenshot 2026-03-12 002711" src="https://github.com/user-attachments/assets/917b02c5-1061-4d51-b94f-5b6a44f42abd" />
<img width="1919" height="979" alt="Screenshot 2026-03-12 002724" src="https://github.com/user-attachments/assets/58339db5-a446-49d8-a001-329442b7c217" />




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



**Wiring:** Both sensors share I²C (SDA: GPIO21, SCL: GPIO22) with 4.7 kΩ pull-ups to 3.3 V.

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



---

## 🔭 Future Work

- [ ] TensorFlow Lite on-device inference (no Wi-Fi needed)
- [ ] Expand to 10 skin disease categories
- [ ] Cloud-hosted HTTPS backend with patient history
- [ ] Custom PCB wearable probe (ESP32 + sensors in 55 mm patch)
- [ ] iOS support via React Native BLE

---

## 📄 Publication

This project was submitted as a B.Tech final year project (Electronics & Computer Science Engineering, KIIT University) and as an IEEE conference paper:

> *"IoT-Integrated Skin Disease Detection and Health Scoring Using MobileNetV2 Deep Learning"*
> 2025 IEEE International Conference on Electronics, Communication and Signal Processing (ICECSP-2025)

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not a medical device and should not be used for clinical diagnosis. Always consult a qualified dermatologist for medical advice.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
