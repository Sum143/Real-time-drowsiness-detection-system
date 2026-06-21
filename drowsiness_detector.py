"""
╔══════════════════════════════════════════════════════════════════╗
║         REAL-TIME DROWSINESS DETECTION SYSTEM                   ║
║         MediaPipe Tasks API (v0.10+) + OpenCV + EAR             ║
╚══════════════════════════════════════════════════════════════════╝
Compatible with mediapipe >= 0.10.0 (new tasks.vision API)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import sys
import platform
import urllib.request
import os

# ─────────────────────────────────────────────
#  CONFIGURATION & THRESHOLDS
# ─────────────────────────────────────────────

EAR_THRESHOLD       = 0.22
EAR_CONSEC_FRAMES   = 20
MAR_THRESHOLD       = 0.65
YAWN_CONSEC_FRAMES  = 15

MODEL_PATH  = "face_landmarker.task"
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Visual palette (BGR)
CLR_ACCENT   = (0,   200, 255)
CLR_GREEN    = (0,   220, 100)
CLR_WARN     = (0,   165, 255)
CLR_DANGER   = (0,   60,  255)
CLR_WHITE    = (240, 240, 245)
CLR_PANEL_BG = (25,  30,  48)

# MediaPipe FaceMesh landmark indices (478-pt with attention mesh)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  291, 39,  181, 0,   17,  269, 405]

# ─────────────────────────────────────────────
#  MODEL DOWNLOAD
# ─────────────────────────────────────────────

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000:
        print(f"[INFO] Model found: {MODEL_PATH}")
        return
    print(f"[INFO] Downloading face landmarker model (~29 MB)...")
    print(f"       URL: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"\n[ERROR] Could not download model: {e}")
        print(f"        Please download manually from:\n        {MODEL_URL}")
        print(f"        and place '{MODEL_PATH}' in the same folder as this script.\n")
        sys.exit(1)

# ─────────────────────────────────────────────
#  MATH HELPERS
# ─────────────────────────────────────────────

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C else 0.0


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices]
    vert  = (euclidean(pts[2], pts[6]) + euclidean(pts[3], pts[7])) / 2
    horiz = euclidean(pts[0], pts[1])
    return vert / horiz if horiz else 0.0

# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────

def draw_eye_landmarks(frame, landmarks, indices, w, h, color):
    pts = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in indices], np.int32)
    cv2.polylines(frame, [pts], True, color, 1)
    for pt in pts:
        cv2.circle(frame, tuple(pt), 2, color, -1)


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i]-c1[i]) * t) for i in range(3))


def draw_ear_bar(frame, ear, x, y, label, alert):
    bar_h, bar_w = 80, 18
    cv2.rectangle(frame, (x, y), (x+bar_w, y+bar_h), (40, 45, 60), -1)
    fill_ratio = min(1.0, ear / 0.4)
    fill_h     = int(bar_h * fill_ratio)
    bar_color  = CLR_DANGER if alert else (CLR_WARN if ear < EAR_THRESHOLD + 0.05 else CLR_GREEN)
    cv2.rectangle(frame, (x, y+bar_h-fill_h), (x+bar_w, y+bar_h), bar_color, -1)
    thr_y = y + bar_h - int(bar_h * (EAR_THRESHOLD / 0.4))
    cv2.line(frame, (x-4, thr_y), (x+bar_w+4, thr_y), CLR_WARN, 1)
    cv2.putText(frame, label,        (x-2, y+bar_h+14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_WHITE, 1)
    cv2.putText(frame, f"{ear:.2f}", (x-2, y-6),        cv2.FONT_HERSHEY_SIMPLEX, 0.35, bar_color, 1)


def draw_hud(frame, fps, ear_left, ear_right, mar, ear_avg,
             closed_ctr, yawn_ctr, state, alert_active,
             s_alerts, s_yawns, elapsed_str):
    fh, fw = frame.shape[:2]
    panel_w = 210
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, fh), CLR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.putText(frame, "DROWSINESS", (10, 28),  cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_ACCENT, 1)
    cv2.putText(frame, "MONITOR",    (10, 50),  cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_ACCENT, 1)
    cv2.line(frame, (10, 56), (panel_w-10, 56), (50, 55, 80), 1)

    cv2.putText(frame, "EYE ASPECT RATIO", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,155,170), 1)
    draw_ear_bar(frame, ear_left,  18, 82, "L", alert_active)
    draw_ear_bar(frame, ear_right, 50, 82, "R", alert_active)

    avg_color = CLR_DANGER if alert_active else (CLR_WARN if ear_avg < EAR_THRESHOLD+0.05 else CLR_GREEN)
    cv2.putText(frame, "AVG",        (85, 95),  cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,155,170), 1)
    cv2.putText(frame, f"{ear_avg:.3f}", (82, 116), cv2.FONT_HERSHEY_DUPLEX, 0.55, avg_color, 1)

    cv2.putText(frame, "CLOSED FRAMES", (10, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,155,170), 1)
    bx, by, bl = 10, 185, panel_w-20
    cv2.rectangle(frame, (bx, by), (bx+bl, by+8), (40,45,60), -1)
    prog = min(1.0, closed_ctr / EAR_CONSEC_FRAMES)
    cv2.rectangle(frame, (bx, by), (bx+int(bl*prog), by+8), lerp_color(CLR_GREEN, CLR_DANGER, prog), -1)
    cv2.putText(frame, f"{closed_ctr}/{EAR_CONSEC_FRAMES}", (bx, by+22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_WHITE, 1)

    cv2.line(frame, (10, 218), (panel_w-10, 218), (50,55,80), 1)
    mar_color = CLR_WARN if mar > MAR_THRESHOLD else CLR_GREEN
    cv2.putText(frame, "MOUTH (YAWN)", (10, 234), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,155,170), 1)
    cv2.putText(frame, f"MAR {mar:.3f}", (10, 253), cv2.FONT_HERSHEY_DUPLEX, 0.45, mar_color, 1)

    cv2.putText(frame, "YAWN FRAMES", (10, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,155,170), 1)
    cv2.rectangle(frame, (bx, 278), (bx+bl, 286), (40,45,60), -1)
    yprog = min(1.0, yawn_ctr / YAWN_CONSEC_FRAMES)
    cv2.rectangle(frame, (bx, 278), (bx+int(bl*yprog), 286), mar_color, -1)
    cv2.putText(frame, f"{yawn_ctr}/{YAWN_CONSEC_FRAMES}", (bx, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_WHITE, 1)

    cv2.line(frame, (10, 315), (panel_w-10, 315), (50,55,80), 1)
    cv2.putText(frame, "SESSION STATS",  (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,155,170), 1)
    cv2.putText(frame, f"Alerts   : {s_alerts}",  (10, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_WHITE, 1)
    cv2.putText(frame, f"Yawns    : {s_yawns}",   (10, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_WHITE, 1)
    cv2.putText(frame, f"Duration : {elapsed_str}",(10, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_WHITE, 1)
    cv2.putText(frame, f"FPS      : {fps:>5.1f}", (10, 399), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_WHITE, 1)

    # Status badge
    badge_x = fw - 180
    bcolor = {"AWAKE": CLR_GREEN, "DROWSY!": CLR_WARN, "ALERT!": CLR_DANGER}.get(state, CLR_GREEN)
    cv2.rectangle(frame, (badge_x, 10), (badge_x+165, 48), bcolor, -1)
    cv2.putText(frame, state, (badge_x+12, 37), cv2.FONT_HERSHEY_DUPLEX, 0.75, (10,12,20), 2)

    # Bottom bar
    cv2.rectangle(frame, (panel_w, fh-28), (fw, fh), (20,22,35), -1)
    cv2.putText(frame,
        f"EAR Threshold: {EAR_THRESHOLD}  |  Consecutive Frames: {EAR_CONSEC_FRAMES}  |  Press Q to quit",
        (panel_w+10, fh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120,125,150), 1)


def draw_alert_overlay(frame, blink_on, alert_type="EYE"):
    if not blink_on:
        return
    fh, fw = frame.shape[:2]
    color = CLR_DANGER if alert_type == "EYE" else CLR_WARN
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), color, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.rectangle(frame, (5, 5), (fw-5, fh-5), color, 3)
    cx = fw // 2
    msg1 = "WARNING: DROWSINESS DETECTED" if alert_type == "EYE" else "WARNING: YAWNING DETECTED"
    cv2.putText(frame, msg1,                (cx-195+2, fh//2-22), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0,0,0),   2)
    cv2.putText(frame, msg1,                (cx-195,   fh//2-22), cv2.FONT_HERSHEY_DUPLEX, 0.85, CLR_WHITE,  2)
    cv2.putText(frame, "WAKE UP! TAKE A BREAK!", (cx-160+2, fh//2+14), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,0,0), 2)
    cv2.putText(frame, "WAKE UP! TAKE A BREAK!", (cx-160,   fh//2+14), cv2.FONT_HERSHEY_DUPLEX, 0.75, color,   2)


def beep_alert():
    if platform.system() == "Windows":
        try:
            import winsound
            winsound.Beep(1000, 300)
        except Exception:
            pass
    else:
        print("\a", end="", flush=True)

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  DROWSINESS DETECTION SYSTEM  — Starting up…")
    print("="*60)

    # Download model if needed
    ensure_model()

    # ── MediaPipe Tasks API (v0.10+) ─────────────────────────────────
    BaseOptions        = mp.tasks.BaseOptions
    FaceLandmarker     = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode  = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    face_landmarker = FaceLandmarker.create_from_options(options)
    print("[INFO] Face Landmarker loaded.")

    # ── Webcam ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        face_landmarker.close()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"[INFO] Camera ready. Press Q to quit.\n")

    # State
    closed_ctr     = 0
    yawn_ctr       = 0
    alert_active   = False
    yawn_alert     = False
    session_alerts = 0
    session_yawns  = 0
    blink_state    = False
    blink_timer    = 0.0
    BLINK_INTERVAL = 0.35

    prev_fps_time  = time.time()
    fps            = 0.0
    frame_count    = 0
    session_start  = time.time()
    timestamp_ms   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # FPS
        frame_count += 1
        now = time.time()
        if now - prev_fps_time >= 0.5:
            fps           = frame_count / (now - prev_fps_time)
            frame_count   = 0
            prev_fps_time = now

        # Session elapsed
        elapsed = int(now - session_start)
        hrs, rem = divmod(elapsed, 3600)
        mins, sc = divmod(rem, 60)
        elapsed_str = f"{hrs:02d}:{mins:02d}:{sc:02d}"

        # ── MediaPipe inference (VIDEO mode needs monotonic timestamps) ──
        timestamp_ms += 33   # ~30 fps
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        ear_left  = 0.0
        ear_right = 0.0
        mar       = 0.0
        ear_avg   = 0.0
        face_found = bool(result.face_landmarks)

        if face_found:
            # result.face_landmarks[0] is a list of NormalizedLandmark objects
            lms = result.face_landmarks[0]

            ear_left  = eye_aspect_ratio(lms, LEFT_EYE,  fw, fh)
            ear_right = eye_aspect_ratio(lms, RIGHT_EYE, fw, fh)
            ear_avg   = (ear_left + ear_right) / 2.0
            mar       = mouth_aspect_ratio(lms, MOUTH, fw, fh)

            eye_color = CLR_DANGER if alert_active else CLR_ACCENT
            draw_eye_landmarks(frame, lms, LEFT_EYE,  fw, fh, eye_color)
            draw_eye_landmarks(frame, lms, RIGHT_EYE, fw, fh, eye_color)

            # Drowsiness counter
            if ear_avg < EAR_THRESHOLD:
                closed_ctr += 1
                if closed_ctr >= EAR_CONSEC_FRAMES:
                    if not alert_active:
                        session_alerts += 1
                        print(f"[ALERT] Drowsiness! Total: {session_alerts}")
                        beep_alert()
                    alert_active = True
            else:
                closed_ctr = max(0, closed_ctr - 1)
                if closed_ctr == 0:
                    alert_active = False

            # Yawn counter
            if mar > MAR_THRESHOLD:
                yawn_ctr += 1
                if yawn_ctr >= YAWN_CONSEC_FRAMES:
                    if not yawn_alert:
                        session_yawns += 1
                        print(f"[YAWN]  Yawn detected! Total: {session_yawns}")
                        beep_alert()
                    yawn_alert = True
            else:
                yawn_ctr = max(0, yawn_ctr - 1)
                if yawn_ctr == 0:
                    yawn_alert = False
        else:
            if closed_ctr > 0:
                closed_ctr -= 1

        # State label
        if alert_active:
            state = "ALERT!"
        elif closed_ctr > EAR_CONSEC_FRAMES // 2:
            state = "DROWSY!"
        else:
            state = "AWAKE"

        # Alert flash
        if alert_active or yawn_alert:
            if now - blink_timer > BLINK_INTERVAL:
                blink_state = not blink_state
                blink_timer = now
            draw_alert_overlay(frame, blink_state, "EYE" if alert_active else "YAWN")
        else:
            blink_state = False

        # HUD
        draw_hud(frame, fps, ear_left, ear_right, mar, ear_avg,
                 closed_ctr, yawn_ctr, state, alert_active,
                 session_alerts, session_yawns, elapsed_str)

        if not face_found:
            cv2.putText(frame, "NO FACE DETECTED", (fw//2-110, fh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, CLR_WARN, 2)

        cv2.imshow("Drowsiness Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()

    print("\n" + "="*60)
    print("  SESSION SUMMARY")
    print(f"  Duration      : {elapsed_str}")
    print(f"  Drowsy Alerts : {session_alerts}")
    print(f"  Yawns Logged  : {session_yawns}")
    print("="*60)


if __name__ == "__main__":
    main()
