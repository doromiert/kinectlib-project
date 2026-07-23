"""
visualizer.py — realtime hand tracking visualizer via browser
open http://localhost:5000 in a browser, or:
  mpv --no-cache --untimed http://localhost:5000/video
"""

import sys, os, threading
sys.path.insert(0, ".")

import cv2
import numpy as np
from flask import Flask, Response
from kinect import Kinect
from hands import HandTracker, RGB_W, RGB_H, DEPTH_W, DEPTH_H, SCALE_X, SCALE_Y

CUTOFF_MARGIN_MM = 150
DISPLAY_W = 960
DISPLAY_H = 540

HAND_COLORS = {
    "Right": (60,  60,  220),
    "Left":  (220, 100, 60),
}

app = Flask(__name__)
_current_frame = None
_lock = threading.Lock()


def depth_to_rgb_coords(depth_frame):
    return cv2.resize(depth_frame, (RGB_W, RGB_H), interpolation=cv2.INTER_NEAREST)


def apply_hand_mask(frame, depth_rgb, hands):
    out = frame.copy()
    for hand in hands:
        _, _, wrist_z = hand["wrist"]
        if wrist_z <= 0:
            continue
        cutoff = wrist_z + CUTOFF_MARGIN_MM
        color  = HAND_COLORS[hand["hand"]]
        behind = depth_rgb > cutoff
        tint = np.zeros_like(out)
        tint[:, :] = color
        out[behind] = (out[behind] * 0.25 + tint[behind] * 0.75).astype(np.uint8)
    return out


def draw_landmarks(frame, hands):
    out = frame.copy()
    for hand in hands:
        color = HAND_COLORS[hand["hand"]]
        for name, (x, y, z) in hand["landmarks"].items():
            cv2.circle(out, (x, y), 6, color, -1)
        for tip in ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]:
            wx, wy, _ = hand["wrist"]
            tx, ty, _ = hand["landmarks"][tip]
            cv2.line(out, (wx, wy), (tx, ty), color, 2)
        wx, wy, wz = hand["wrist"]
        ix, iy, iz = hand["index_tip"]
        source = hand.get("source", "rgb")
        label = f"{hand['hand']} [{source}]  wrist:{wz:.0f}mm  index:{iz:.0f}mm"
        cv2.putText(out, label, (wx - 20, wy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def gen_frames():
    while True:
        with _lock:
            if _current_frame is None:
                continue
            _, buf = cv2.imencode('.jpg', _current_frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buf.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + frame_bytes + b'\r\n')


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return '''
    <html><body style="margin:0;background:#000">
    <img src="/video" style="width:100%;height:100vh;object-fit:contain">
    </body></html>
    '''


def main():
    global _current_frame

    print("opening kinect...")
    with Kinect() as kinect, HandTracker() as tracker:
        print("warming up...")
        for _ in range(30):
            kinect.get_frames()

        t = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True),
            daemon=True
        )
        t.start()
        print("open http://localhost:5000 or mpv --no-cache --untimed http://localhost:5000/video")
        print("ctrl+c to quit")

        while True:
            rgb, depth, ir = kinect.get_frames()
            if rgb is None:
                continue

            hands = tracker.process(rgb, depth, ir)
            depth_rgb = depth_to_rgb_coords(depth)
            frame = apply_hand_mask(rgb[:, :, :3], depth_rgb, hands)
            frame = draw_landmarks(frame, hands)
            frame_small = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

            with _lock:
                _current_frame = frame_small


if __name__ == "__main__":
    main()
