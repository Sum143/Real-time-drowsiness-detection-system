# 😴 Real-Time Drowsiness Detection System

> Driver safety through facial landmark analysis — MediaPipe + OpenCV + EAR Algorithm

---

## 📋 Project Overview

This system monitors eye movements through a webcam and detects fatigue in real time using **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** computed from 468 facial landmarks via Google MediaPipe Face Mesh.

---

## 🔍 Workflow

```
📹 Webcam Input
      ↓
🧠 Face Detection (MediaPipe FaceMesh — 468 landmarks)
      ↓
👁️  Eye Landmark Extraction (6 pts per eye)
      ↓
📐 EAR Calculation  +  MAR (yawn) Calculation
      ↓
⏱️  Consecutive Frame Counting
      ↓
🚨 Alert Generation (visual flash + beep)
```

---

## 🧮 EAR Formula — Soukupova & Cech (2016)

```
        ||p2−p6|| + ||p3−p5||
EAR = ─────────────────────────
           2 × ||p1−p4||
```

| EAR Value   | Meaning        |
|-------------|----------------|
| 0.25 – 0.40 | Eyes open      |
| 0.15 – 0.22 | Eyes closing   |
| < 0.22      | Eyes shut      |

**Alert triggered** when EAR < 0.22 for ≥ 20 consecutive frames.

---

## 🗂️ Project Structure

```
drowsiness_detection/
├── drowsiness_detector.py   # Main detection script
├── requirements.txt         # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### Step 1 — Create virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run
```bash
python drowsiness_detector.py
```
Press **Q** to quit.

---

## 🎛️ Configurable Parameters (top of script)

| Parameter           | Default | Description                              |
|---------------------|---------|------------------------------------------|
| `EAR_THRESHOLD`     | `0.22`  | EAR below this = eye closed             |
| `EAR_CONSEC_FRAMES` | `20`    | Frames before drowsiness alert fires    |
| `MAR_THRESHOLD`     | `0.65`  | MAR above this = yawning               |
| `YAWN_CONSEC_FRAMES`| `15`    | Frames before yawn alert fires          |

---

## 🖥️ HUD Features

- **Left panel** — Live EAR bars (L/R), avg EAR, closed-frame progress bar, MAR gauge, session stats
- **Status badge** — AWAKE / DROWSY! / ALERT! in colour-coded indicator
- **Alert overlay** — Full-screen red/orange flash with message when drowsiness detected
- **Session summary** — Printed to terminal on exit (duration, alerts, yawns)

---

## 🛠️ Tech Stack

| Tool        | Purpose                        |
|-------------|--------------------------------|
| Python 3.9+ | Core language                  |
| OpenCV      | Webcam capture, drawing, display |
| MediaPipe   | 468-point face mesh detection  |
| NumPy       | Landmark coordinate math       |

---

## 👤 Author

**Suman Prasad Gouda**  
Data Science Intern — Innomatics Research Labs, Hyderabad
