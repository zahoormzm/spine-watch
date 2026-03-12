#!/usr/bin/env python3
"""
PostureGuard — Real-Time Privacy-First Posture Tracking
Target: Raspberry Pi 5 · Pi Camera V2

Controls:  q / ESC = quit     +/- = adjust threshold

Privacy: ALL processing happens in RAM. Zero frames touch disk. Ever.
"""

import argparse
import math
import time
from collections import deque
from contextlib import contextmanager

import cv2
import mediapipe as mp
import numpy as np

DEFAULT_THRESHOLD = 140
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
FPS_WINDOW        = 30
SLOUCH_HOLD       = 8
ANGLE_SMOOTH      = 5

EMERALD  = (80,  200, 100)
CORAL    = (50,   60, 220)
AMBER    = (30,  180, 255)
WHITE    = (255, 255, 255)
PANEL_BG = (20,   20,  20)

_P = mp.solutions.pose.PoseLandmark
LM = {
    "left":  (_P.LEFT_EAR,  _P.LEFT_SHOULDER,  _P.LEFT_HIP),
    "right": (_P.RIGHT_EAR, _P.RIGHT_SHOULDER, _P.RIGHT_HIP),
}


def calc_angle(a: tuple, b: tuple, c: tuple) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    denom = math.hypot(*ba) * math.hypot(*bc)
    if denom < 1e-6:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (ba[0]*bc[0] + ba[1]*bc[1]) / denom))))


def get_landmark_px(landmarks, lid, w: int, h: int):
    lm = landmarks[lid]
    if lm.visibility < 0.7:
        return None
    x, y = int(lm.x * w), int(lm.y * h)
    if not (0 <= x <= w and 0 <= y <= h):
        return None
    return x, y, lm.visibility


def best_posture_angle(landmarks, w: int, h: int) -> tuple:
    best = (None, None, None, -1.0)
    for side, (eid, sid, hid) in LM.items():
        ear = get_landmark_px(landmarks, eid, w, h)
        sho = get_landmark_px(landmarks, sid, w, h)
        hip = get_landmark_px(landmarks, hid, w, h)
        if not (ear and sho and hip):
            continue
        if hip[1] <= sho[1] or (hip[1] - sho[1]) > h:
            continue
        vis = (ear[2] + sho[2] + hip[2]) / 3.0
        if vis > best[3]:
            best = (
                calc_angle(ear[:2], sho[:2], hip[:2]),
                side,
                {"ear": ear[:2], "shoulder": sho[:2], "hip": hip[:2]},
                vis,
            )
    return best[:3]


def _rounded_fill(img, p1: tuple, p2: tuple, color: tuple, r: int = 10):
    x1, y1 = p1
    x2, y2 = p2
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, -1)


def draw_hud(frame, angle, threshold: float, is_slouching: bool,
             session_secs: float, detected: bool):
    h, w = frame.shape[:2]
    F = cv2.FONT_HERSHEY_SIMPLEX

    if detected:
        accent = CORAL if is_slouching else EMERALD
        text   = ("SLOUCHING" if is_slouching else "POSTURE OK") + (
            f"   {angle:.0f}\u00b0" if angle is not None else ""
        )
    else:
        accent = AMBER
        text   = "NO PERSON DETECTED"

    (tw, th), _ = cv2.getTextSize(text, F, 0.65, 2)
    px, py = 18, 9
    bx1 = (w - tw) // 2 - px;  by1 = 10
    bx2 = (w + tw) // 2 + px;  by2 = by1 + th + 2 * py

    ov = frame.copy()
    _rounded_fill(ov, (bx1, by1), (bx2, by2), PANEL_BG, r=12)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (bx1 + 2, by1 + 5), (bx1 + 5, by2 - 5), accent, -1)
    cv2.putText(frame, text, ((w - tw) // 2, by1 + py + th),
                F, 0.65, accent, 2, cv2.LINE_AA)

    strip_h = 26
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - strip_h), (w, h), PANEL_BG, -1)
    cv2.addWeighted(ov2, 0.72, frame, 0.28, 0, frame)

    m, s  = divmod(int(session_secs), 60)
    hr, m = divmod(m, 60)
    info  = f"{hr:02d}:{m:02d}:{s:02d}   \u00b7   Threshold: {threshold:.0f}\u00b0   ( +/- )"
    (iw, _), _ = cv2.getTextSize(info, F, 0.42, 1)
    cv2.putText(frame, info, ((w - iw) // 2, h - 8), F, 0.42, AMBER, 1, cv2.LINE_AA)


def draw_angle_arc(frame, pts: dict, color: tuple, r: int = 40):
    if not pts:
        return
    sho, ear, hip = pts["shoulder"], pts["ear"], pts["hip"]
    a1 = math.degrees(math.atan2(ear[1] - sho[1], ear[0] - sho[0]))
    a2 = math.degrees(math.atan2(hip[1] - sho[1], hip[0] - sho[0]))
    cv2.ellipse(frame, sho, (r, r), 0, min(a1, a2), max(a1, a2), color, 2, cv2.LINE_AA)


def draw_key_points(frame, pts: dict, color: tuple):
    if not pts:
        return
    for x, y in pts.values():
        cv2.circle(frame, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 7, WHITE,  1, cv2.LINE_AA)


@contextmanager
def open_picamera(width: int, height: int):
    from picamera2 import Picamera2
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"format": "RGB888", "size": (width, height)},
        raw={"size": (1640, 1232)},  # full-FOV 2x2 binned — prevents centre crop
        buffer_count=4,
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)  # let auto-exposure settle
    print(f"[Camera] Streaming {width}x{height} RGB888")
    try:
        yield cam
    finally:
        cam.stop()
        cam.close()
        print("[Camera] Released.")


def main():
    ap = argparse.ArgumentParser(description="PostureGuard — Pi 5 Posture Tracker")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--width",    type=int,   default=FRAME_WIDTH)
    ap.add_argument("--height",   type=int,   default=FRAME_HEIGHT)
    ap.add_argument("--model",    type=int,   default=1, choices=[0, 1, 2])
    ap.add_argument("--no-flip",  action="store_true")
    args = ap.parse_args()

    threshold  = args.threshold
    mirror     = not args.no_flip
    fps_times  = deque(maxlen=FPS_WINDOW)
    angle_buf  = deque(maxlen=ANGLE_SMOOTH)
    slouch_ctr = 0
    session_start = time.time()

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    WIN = "PostureGuard"

    try:
        with open_picamera(args.width, args.height) as cam:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, args.width, args.height)
            print(f"[PostureGuard] Running — threshold {threshold:.0f}° | q/ESC to quit")

            while True:
                t0    = time.perf_counter()
                frame = cam.capture_array()
                if mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False  # perf hint for MediaPipe
                results = pose.process(rgb)
                rgb.flags.writeable = True

                angle = side = pts = None
                is_slouch = False
                detected  = False

                if results.pose_landmarks:
                    detected = True
                    lms      = results.pose_landmarks.landmark
                    angle, side, pts = best_posture_angle(lms, w, h)

                    if angle is not None:
                        angle_buf.append(angle)
                        angle = sum(angle_buf) / len(angle_buf)
                        if angle < threshold:
                            slouch_ctr = min(slouch_ctr + 1, SLOUCH_HOLD + 1)
                        else:
                            slouch_ctr = max(slouch_ctr - 2, 0)
                        is_slouch = slouch_ctr >= SLOUCH_HOLD

                    sk   = CORAL if is_slouch else EMERALD
                    spec = mp_draw.DrawingSpec(color=sk, thickness=2, circle_radius=2)
                    mp_draw.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=spec, connection_drawing_spec=spec,
                    )
                    draw_key_points(frame, pts, sk)
                    draw_angle_arc(frame, pts, sk)

                fps_times.append(time.perf_counter() - t0)

                draw_hud(frame, angle, threshold, is_slouch,
                         time.time() - session_start, detected)

                cv2.imshow(WIN, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key in (ord("+"), ord("=")):
                    threshold = min(threshold + 5, 180)
                    print(f"[PostureGuard] Threshold → {threshold:.0f}°")
                elif key in (ord("-"), ord("_")):
                    threshold = max(threshold - 5, 90)
                    print(f"[PostureGuard] Threshold → {threshold:.0f}°")

    except KeyboardInterrupt:
        print("\n[PostureGuard] Interrupted.")
    except ImportError as e:
        print(f"\n[Error] Missing dependency: {e}")
        import sys; sys.exit(1)
    finally:
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
