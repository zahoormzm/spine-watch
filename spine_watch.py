#!/usr/bin/env python3
"""
SpineWatch — Real-Time Privacy-First Posture Tracking
Target: Raspberry Pi 5 · Pi Camera V2

Controls:  q / ESC = quit     +/- = adjust threshold     c = calibrate

Privacy: ALL processing happens in RAM. Zero frames touch disk. Ever.
"""

import argparse
import io
import math
import os
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from contextlib import contextmanager

import cv2
import mediapipe as mp
import numpy as np

DEFAULT_THRESHOLD = 5.0    # degrees of forward head tilt from calibrated upright
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
FPS_WINDOW        = 30
SLOUCH_HOLD       = 10     # frames of sustained slouch before triggering
ANGLE_SMOOTH      = 11     # smoothing window (odd = better median)
EMA_ALPHA         = 0.35   # exponential moving average weight (lower = smoother)
VIS_THRESHOLD     = 0.5    # minimum landmark visibility
BUZZER_PIN        = 18     # BCM GPIO pin

# (min_continuous_seconds, cooldown_seconds, alert_style)
ALERT_LEVELS = [
    ( 0, 2.5, "gentle"),
    ( 2, 2.0, "nudge"),
    ( 4, 1.5, "warning"),
    ( 7, 1.0, "urgent"),
    (10, 0.8, "alarm"),
]

# BGR colors for OpenCV
EMERALD  = (120, 215, 100)
CORAL    = (80,   80, 230)
AMBER    = (50,  190, 255)
WHITE    = (240, 240, 240)
GRAY     = (160, 160, 160)
PANEL_BG = (30,   30,  35)
DARK_BG  = (18,   18,  22)

_P = mp.solutions.pose.PoseLandmark


def get_alert_params(continuous_secs: float) -> tuple:
    cooldown, style = ALERT_LEVELS[0][1:]
    for min_secs, cd, s in ALERT_LEVELS:
        if continuous_secs >= min_secs:
            cooldown, style = cd, s
    return cooldown, style


def _make_tone(freq, dur, sr=44100, vol=0.5):
    """Generate a sine tone with smooth fade-in/out to avoid clicks."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    # Apply fade envelope (20ms fade in/out)
    fade = int(sr * 0.02)
    if fade > 0 and len(tone) > 2 * fade:
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
    return (tone * vol * 32767).astype(np.int16)


def _build_alert(style: str, sr: int = 44100):
    """Build alert audio based on escalation style."""
    silence = lambda ms: np.zeros(int(sr * ms / 1000), dtype=np.int16)

    if style == "gentle":
        # Soft two-note chime (C6 → E6), like a doorbell
        return np.concatenate([
            _make_tone(1047, 0.15, sr, 0.3),
            silence(50),
            _make_tone(1319, 0.25, sr, 0.3),
        ])

    elif style == "nudge":
        # Three rising notes (E5 → G5 → B5)
        return np.concatenate([
            _make_tone(659, 0.12, sr, 0.4),
            silence(40),
            _make_tone(784, 0.12, sr, 0.4),
            silence(40),
            _make_tone(988, 0.2, sr, 0.4),
        ])

    elif style == "warning":
        # Two-tone alert repeated twice (like a notification)
        pair = np.concatenate([
            _make_tone(880, 0.15, sr, 0.5),
            silence(60),
            _make_tone(1109, 0.15, sr, 0.5),
        ])
        return np.concatenate([pair, silence(150), pair])

    elif style == "urgent":
        # Fast descending triplets (alarm feel)
        notes = []
        for freq in [1480, 1175, 880, 1480, 1175, 880]:
            notes.append(_make_tone(freq, 0.1, sr, 0.6))
            notes.append(silence(30))
        return np.concatenate(notes)

    else:  # alarm
        # Rapid siren sweep
        t = np.linspace(0, 1.0, sr)
        sweep_freq = 800 + 600 * np.sin(2 * np.pi * 4 * t)  # 4Hz wobble
        phase = np.cumsum(sweep_freq / sr)
        siren = (np.sin(2 * np.pi * phase) * 0.7 * 32767).astype(np.int16)
        return siren


_alert_cache = {}


def _get_alert_wav(style: str) -> bytes:
    """Return cached WAV bytes for an alert style."""
    if style not in _alert_cache:
        sr = 44100
        audio = _build_alert(style, sr)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        _alert_cache[style] = buf.getvalue()
    return _alert_cache[style]


def _play_sound(style: str):
    import tempfile
    wav_data = _get_alert_wav(style)
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(wav_data)
        tmp.close()
        subprocess.Popen(
            ["pw-play", tmp.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).wait()
        os.unlink(tmp.name)
    except FileNotFoundError:
        pass


def play_alert(style: str = "gentle"):
    threading.Thread(target=_play_sound, args=(style,), daemon=True).start()


def _buzz_worker(buzzer, n: int):
    try:
        buzzer.on()
    except Exception:
        pass


def init_buzzer(pin: int):
    try:
        from gpiozero import Buzzer as GZBuzzer
        bz = GZBuzzer(pin)
        print(f"[Buzzer] Initialised on GPIO {pin} (BCM)")
        return bz
    except Exception as e:
        print(f"[Buzzer] Could not initialise on GPIO {pin}: {e}")
        return None


def buzz_alert(buzzer, n: int = 1):
    if buzzer is not None:
        threading.Thread(target=_buzz_worker, args=(buzzer, n), daemon=True).start()


def get_lm(landmarks, lid, w: int, h: int):
    lm = landmarks[lid]
    if lm.visibility < VIS_THRESHOLD:
        return None
    x, y = lm.x * w, lm.y * h
    if not (0 <= x <= w and 0 <= y <= h):
        return None
    return (x, y, lm.visibility)


def compute_neck_angle(landmarks, w: int, h: int) -> tuple:
    """
    Measure forward head position using the angle between:
      - vertical line through mid-shoulder
      - line from mid-shoulder to mid-ear (or nose)

    Returns (angle_from_vertical, keypoints_dict) or (None, None).
    A perfectly upright posture = ~0 degrees.
    Slouching forward = larger positive angle.
    """
    l_sho = get_lm(landmarks, _P.LEFT_SHOULDER, w, h)
    r_sho = get_lm(landmarks, _P.RIGHT_SHOULDER, w, h)

    if not l_sho or not r_sho:
        return None, None

    mid_sho = ((l_sho[0] + r_sho[0]) / 2, (l_sho[1] + r_sho[1]) / 2)

    # Try nose first (most reliably detected), fall back to mid-ear
    nose = get_lm(landmarks, _P.NOSE, w, h)
    l_ear = get_lm(landmarks, _P.LEFT_EAR, w, h)
    r_ear = get_lm(landmarks, _P.RIGHT_EAR, w, h)

    if nose:
        head = (nose[0], nose[1])
    elif l_ear and r_ear:
        head = ((l_ear[0] + r_ear[0]) / 2, (l_ear[1] + r_ear[1]) / 2)
    elif l_ear:
        head = (l_ear[0], l_ear[1])
    elif r_ear:
        head = (r_ear[0], r_ear[1])
    else:
        return None, None

    # Also get hip midpoint for drawing
    l_hip = get_lm(landmarks, _P.LEFT_HIP, w, h)
    r_hip = get_lm(landmarks, _P.RIGHT_HIP, w, h)
    mid_hip = None
    if l_hip and r_hip:
        mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
    elif l_hip:
        mid_hip = (l_hip[0], l_hip[1])
    elif r_hip:
        mid_hip = (r_hip[0], r_hip[1])

    # Angle from vertical: atan2(horizontal_offset, vertical_offset)
    dx = head[0] - mid_sho[0]
    dy = mid_sho[1] - head[1]  # positive = head above shoulder (normal)

    if dy < 1:  # head not above shoulders — bad detection
        return None, None

    angle_from_vertical = math.degrees(math.atan2(abs(dx), dy))

    pts = {
        "head": (int(head[0]), int(head[1])),
        "mid_shoulder": (int(mid_sho[0]), int(mid_sho[1])),
        "l_shoulder": (int(l_sho[0]), int(l_sho[1])),
        "r_shoulder": (int(r_sho[0]), int(r_sho[1])),
    }
    if mid_hip:
        pts["mid_hip"] = (int(mid_hip[0]), int(mid_hip[1]))

    return angle_from_vertical, pts


def _rounded_fill(img, p1: tuple, p2: tuple, color: tuple, r: int = 10):
    x1, y1 = p1
    x2, y2 = p2
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, -1)


def _draw_pill(frame, center_x: int, y: int, text: str, color: tuple,
               font=cv2.FONT_HERSHEY_SIMPLEX, scale: float = 0.55, thickness: int = 2):
    """Draw text inside a rounded pill-shaped badge."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    px, py = 14, 8
    x1 = center_x - tw // 2 - px
    y1 = y - py
    x2 = center_x + tw // 2 + px
    y2 = y + th + py
    r = (y2 - y1) // 2

    ov = frame.copy()
    _rounded_fill(ov, (x1, y1), (x2, y2), PANEL_BG, r=r)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    # Accent bar on left edge
    cv2.rectangle(frame, (x1 + 3, y1 + 4), (x1 + 5, y2 - 4), color, -1)
    cv2.putText(frame, text, (center_x - tw // 2, y + th),
                font, scale, color, thickness, cv2.LINE_AA)
    return x1, y1, x2, y2


def draw_hud(frame, angle, threshold: float, is_slouching: bool,
             session_secs: float, detected: bool, calibrated: bool,
             fps: float = 0.0, posture_pct: float = -1.0, paused: bool = False):
    h, w = frame.shape[:2]
    F = cv2.FONT_HERSHEY_SIMPLEX

    # --- Top status pill ---
    if paused:
        accent = AMBER
        label  = "PAUSED"
    elif detected:
        accent = CORAL if is_slouching else EMERALD
        label  = "SLOUCHING" if is_slouching else "GOOD POSTURE"
    else:
        accent = AMBER
        label  = "NO PERSON"

    if angle is not None and not paused:
        status_text = f"{label}  {angle:.1f}\u00b0"
    else:
        status_text = label
    _draw_pill(frame, w // 2, 14, status_text, accent, F, 0.58, 2)

    # --- Top-left: posture score ---
    if posture_pct >= 0:
        score_color = EMERALD if posture_pct >= 80 else (AMBER if posture_pct >= 50 else CORAL)
        score_text = f"{posture_pct:.0f}%"
        ov = frame.copy()
        _rounded_fill(ov, (8, 8), (62, 36), DARK_BG, r=10)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, score_text, (14, 30), F, 0.45, score_color, 1, cv2.LINE_AA)

    # --- Top-right: FPS ---
    if fps > 0:
        fps_text = f"{fps:.0f}"
        ov = frame.copy()
        _rounded_fill(ov, (w - 58, 8), (w - 8, 36), DARK_BG, r=10)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, fps_text, (w - 50, 30), F, 0.42, GRAY, 1, cv2.LINE_AA)

    # --- Bottom bar ---
    strip_h = 30
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - strip_h), (w, h), DARK_BG, -1)
    cv2.addWeighted(ov2, 0.78, frame, 0.22, 0, frame)

    m, s  = divmod(int(session_secs), 60)
    hr, m = divmod(m, 60)
    cal_icon = "\u2713" if calibrated else "!"
    time_str = f"{hr:02d}:{m:02d}:{s:02d}"
    info = f"{time_str}   Thresh {threshold:.0f}\u00b0   [{cal_icon}]   +  -  c  p  q"
    (iw, _), _ = cv2.getTextSize(info, F, 0.38, 1)
    cv2.putText(frame, info, ((w - iw) // 2, h - 10), F, 0.38, GRAY, 1, cv2.LINE_AA)


def draw_posture_overlay(frame, pts: dict, color: tuple):
    if not pts:
        return
    mid_sho = pts.get("mid_shoulder")
    head = pts.get("head")
    mid_hip = pts.get("mid_hip")
    l_sho = pts.get("l_shoulder")
    r_sho = pts.get("r_shoulder")

    # Semi-transparent lines on overlay for a softer look
    ov = frame.copy()

    # Draw shoulder line
    if l_sho and r_sho:
        cv2.line(ov, l_sho, r_sho, color, 3, cv2.LINE_AA)

    # Draw spine line (mid_shoulder to mid_hip)
    if mid_sho and mid_hip:
        cv2.line(ov, mid_sho, mid_hip, color, 3, cv2.LINE_AA)

    # Draw neck line (mid_shoulder to head)
    if mid_sho and head:
        cv2.line(ov, mid_sho, head, color, 3, cv2.LINE_AA)

    # Draw vertical reference (dashed feel via thinner line)
    if mid_sho:
        vert_top = (mid_sho[0], mid_sho[1] - 90)
        cv2.line(ov, mid_sho, vert_top, GRAY, 1, cv2.LINE_AA)

    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

    # Draw keypoints on top (not blended)
    for name, pt in pts.items():
        r = 9 if name == "head" else 6
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r + 1, WHITE, 1, cv2.LINE_AA)


def print_summary(start: float, good: int, bad: int, events: int,
                  asum: float, acount: int):
    dur     = time.time() - start
    m, s    = divmod(int(dur), 60)
    hr, m   = divmod(m, 60)
    total   = good + bad
    pct     = good / total * 100 if total else 0.0
    avg_ang = asum  / acount     if acount else 0.0
    W = 40

    def row(label, value):
        line = f"  {label:<22}{value}"
        print(f"\u2551{line:<{W}}\u2551")

    print(f"\n\u2554{'═'*W}\u2557")
    title = "  PostureGuard \u2014 Session Summary"
    print(f"\u2551{title:<{W}}\u2551")
    print(f"\u2560{'═'*W}\u2563")
    row("Duration:",        f"{hr:02d}:{m:02d}:{s:02d}")
    row("Good posture:",    f"{pct:.1f}%")
    row("Slouch episodes:", str(events))
    row("Average offset:",  f"{avg_ang:.1f}\u00b0")
    print(f"\u255a{'═'*W}\u255d\n")


@contextmanager
def open_picamera(width: int, height: int):
    from picamera2 import Picamera2
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"format": "RGB888", "size": (width, height)},
        raw={"size": (1640, 1232)},
        buffer_count=4,
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)
    print(f"[Camera] Streaming {width}x{height} RGB888")
    try:
        yield cam
    finally:
        cam.stop()
        cam.close()
        print("[Camera] Released.")


def main():
    ap = argparse.ArgumentParser(description="PostureGuard — Pi 5 Posture Tracker")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Slouch angle offset in degrees (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--width",      type=int,  default=FRAME_WIDTH)
    ap.add_argument("--height",     type=int,  default=FRAME_HEIGHT)
    ap.add_argument("--model",      type=int,  default=1, choices=[0, 1, 2],
                    help="MediaPipe complexity: 0=lite 1=full 2=heavy (default 1)")
    ap.add_argument("--no-flip",    action="store_true",
                    help="Disable horizontal mirror mode")
    ap.add_argument("--buzzer",     action="store_true",
                    help="Enable GPIO buzzer alert")
    ap.add_argument("--buzzer-pin", type=int, default=BUZZER_PIN,
                    help=f"BCM GPIO pin for the buzzer (default {BUZZER_PIN})")
    args = ap.parse_args()

    threshold = args.threshold
    mirror    = not args.no_flip
    buzzer    = init_buzzer(args.buzzer_pin) if args.buzzer else None

    angle_buf      = deque(maxlen=ANGLE_SMOOTH)
    slouch_ctr     = 0
    session_start  = time.time()
    good_frames    = 0
    bad_frames     = 0
    slouch_events  = 0
    last_slouching = False
    angle_sum      = 0.0
    angle_count    = 0
    last_alert_t   = 0.0
    slouch_start_t = None
    calibration_offset = 0.0
    calibrated = False
    ema_angle = None
    paused = False
    fps_times      = deque(maxlen=FPS_WINDOW)

    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    WIN = "PostureGuard"

    try:
        with open_picamera(args.width, args.height) as cam:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, args.width, args.height)
            print(f"[PostureGuard] Running  — threshold {threshold:.0f} degrees")
            print(f"[PostureGuard] Controls: q/ESC=quit  +/-=threshold  c=calibrate  p=pause")
            print(f"[PostureGuard] Sit up straight and press 'c' to calibrate!")

            while True:
                now_t = time.time()
                fps_times.append(now_t)
                if len(fps_times) >= 2:
                    fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                else:
                    fps = 0.0

                frame = cam.capture_array()
                if mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                angle = None
                pts = None
                is_slouch = False
                detected  = False

                if paused:
                    total = good_frames + bad_frames
                    posture_pct = good_frames / total * 100 if total else -1.0
                    draw_hud(frame, angle, threshold, is_slouch,
                             time.time() - session_start, detected, calibrated,
                             fps, posture_pct, paused=True)
                    cv2.imshow(WIN, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("p"):
                        paused = False
                        print("[PostureGuard] Resumed")
                    continue

                # PiCamera2 RGB888 on PiSP actually delivers BGR
                # Convert to RGB for MediaPipe, keep original BGR for display
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)

                if results.pose_landmarks:
                    detected = True
                    lms = results.pose_landmarks.landmark
                    raw_angle, pts = compute_neck_angle(lms, w, h)

                    if raw_angle is not None:
                        # Apply calibration: subtract the "good posture" baseline
                        corrected = raw_angle - calibration_offset
                        angle_buf.append(corrected)

                        # Use median for smoothing (rejects outliers better than mean)
                        sorted_buf = sorted(angle_buf)
                        median_angle = sorted_buf[len(sorted_buf) // 2]

                        # Layer EMA on top of median to dampen jitter
                        if ema_angle is None:
                            ema_angle = median_angle
                        else:
                            ema_angle += EMA_ALPHA * (median_angle - ema_angle)
                        angle = ema_angle

                        if angle > threshold:
                            slouch_ctr = min(slouch_ctr + 1, SLOUCH_HOLD + 2)
                        else:
                            slouch_ctr = max(slouch_ctr - 2, 0)
                        is_slouch = slouch_ctr >= SLOUCH_HOLD

                        if is_slouch:
                            bad_frames += 1
                        else:
                            good_frames += 1
                        angle_sum   += angle
                        angle_count += 1

                        if is_slouch and not last_slouching:
                            slouch_events += 1
                            slouch_start_t = time.time()
                        elif not is_slouch:
                            slouch_start_t = None
                            if buzzer is not None:
                                buzzer.off()
                        last_slouching = is_slouch

                        now = time.time()
                        if is_slouch and slouch_start_t is not None:
                            cooldown, style = get_alert_params(now - slouch_start_t)
                            if (now - last_alert_t) >= cooldown:
                                play_alert(style)
                                buzz_alert(buzzer, 1)
                                last_alert_t = now

                    sk = CORAL if is_slouch else EMERALD
                    draw_posture_overlay(frame, pts, sk)

                total = good_frames + bad_frames
                posture_pct = good_frames / total * 100 if total else -1.0
                draw_hud(frame, angle, threshold, is_slouch,
                         time.time() - session_start, detected, calibrated,
                         fps, posture_pct)

                cv2.imshow(WIN, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key in (ord("+"), ord("=")):
                    threshold = min(threshold + 2, 60)
                    print(f"[PostureGuard] Threshold → {threshold:.0f}")
                elif key in (ord("-"), ord("_")):
                    threshold = max(threshold - 2, 0)
                    print(f"[PostureGuard] Threshold → {threshold:.0f}")
                elif key == ord("p"):
                    paused = True
                    print("[PostureGuard] Paused — press 'p' to resume")
                elif key == ord("c"):
                    if results.pose_landmarks:
                        raw_angle, _ = compute_neck_angle(
                            results.pose_landmarks.landmark, w, h)
                        if raw_angle is not None:
                            calibration_offset = raw_angle
                            calibrated = True
                            angle_buf.clear()
                            ema_angle = None
                            slouch_ctr = 0
                            print(f"[PostureGuard] Calibrated! Baseline: {raw_angle:.1f} degrees")
                        else:
                            print("[PostureGuard] Calibration failed — landmarks not clear")
                    else:
                        print("[PostureGuard] Calibration failed — no person detected")

    except KeyboardInterrupt:
        print("\n[PostureGuard] Interrupted.")
    except ImportError as e:
        print(f"\n[Error] Missing dependency: {e}")
        sys.exit(1)
    finally:
        pose.close()
        cv2.destroyAllWindows()
        if buzzer is not None:
            buzzer.close()
            print("[Buzzer] Released.")
        print_summary(session_start, good_frames, bad_frames,
                      slouch_events, angle_sum, angle_count)


if __name__ == "__main__":
    main()
