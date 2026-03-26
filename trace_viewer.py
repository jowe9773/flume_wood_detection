"""
Paste this function into your existing script and call it at the end like:

    view_traces(df, video_path)

or with optional arguments:

    view_traces(df, video_path, trail_len=50, scale=0.8)

Requirements:
    pip install opencv-python

Expected DataFrame columns:
    track_id, frame, center_x, center_y, width, height, class_name (optional)

Controls:
    SPACE       play / pause
    LEFT / A    step back 1 frame
    RIGHT / D   step forward 1 frame
    + / -       increase / decrease trail length
    S           save current frame as PNG
    Q / ESC     quit
"""

import cv2
import numpy as np
from collections import defaultdict


def view_traces(df, video_path: str, output_path: str, start_frame: int = 0, duration: int =24*30, trail_len: int = 100, scale: float = 0.6, box_thickness: float = 4, font_scale: float = 0.8, font_thickness: float = 2, trail_thickness: float = 4):

    """
    Overlay tracking traces onto a video and display in a resizable window.

    Args:
        df:          pandas DataFrame with columns:
                     track_id, frame, center_x, center_y, width, height
                     (class_name is optional but will be shown if present)
        video_path:  path to the MP4 video file
        trail_len:   number of past frames to draw as a fading trail (default 30)
        scale:       initial window size as fraction of video resolution (default 0.6)
    """

    # ── validate columns ─────────────────────────────────────────────────────
    required = {"track_id", "frame", "center_x", "center_y", "width", "height"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    has_label = "class_name" in df.columns

    # ── build lookup: data[track_id][frame] = dict ───────────────────────────
    data = defaultdict(dict)
    for row in df.itertuples(index=False):
        data[int(row.track_id)][int(row.frame)] = {
            "cx": float(row.center_x),
            "cy": float(row.center_y),
            "w":  float(row.width),
            "h":  float(row.height),
            "label": str(getattr(row, "class_name", "")) if has_label else "",
        }

    track_ids   = sorted(data.keys())
    track_frames = {tid: sorted(data[tid].keys()) for tid in track_ids}

    # ── colour palette (HSV → BGR, one colour per track_id) ──────────────────
    def _make_colour(i, n):
        hue = int(180 * i / max(n, 1))
        hsv = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    id_colour = {tid: _make_colour(i, len(track_ids)) for i, tid in enumerate(track_ids)}

    # ── open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay_ms     = max(1, int(1000 / fps))

    

    win = "Trace Viewer  [SPACE=play/pause  ←/→=step  +/-=trail  S=save  Q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(vid_w * scale), int(vid_h * scale))

    # ── video writer ───────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (vid_w, vid_h))

    # ── inner helpers ─────────────────────────────────────────────────────────
    def _draw_box(img, cx, cy, w, h, colour, tid, label):
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, box_thickness)

        text = f"id:{tid}" + (f" {label}" if label else "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        tx, ty = x1, max(y1 - 4, th + 4)
        cv2.rectangle(img, (tx, ty - th - 3), (tx + tw + 2, ty + 2), colour, -1)
        cv2.putText(img, text, (tx + 1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    def _draw_trail(img, pts, colour, tlen):
        pts = pts[-tlen:]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thickness = int(2 + trail_thickness * alpha)  # grows toward present
            c = tuple(int(v * alpha) for v in colour)
            p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            p2 = (int(pts[i][0]),     int(pts[i][1]))
            cv2.line(img, p1, p2, c, thickness, cv2.LINE_AA)

    def _fit(frame_img, ww, wh):
        fh, fw = frame_img.shape[:2]
        s = min(ww / fw, wh / fh)
        return cv2.resize(frame_img, (int(fw * s), int(fh * s)),
                          interpolation=cv2.INTER_LINEAR)

    def _render(frame_idx, tlen, playing):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        if not ret:
            return None

        for tid in track_ids:
            colour = id_colour[tid]

            # trail: all centres up to this frame
            pts = [(data[tid][f]["cx"], data[tid][f]["cy"])
                   for f in track_frames[tid] if f <= frame_idx]
            if len(pts) > 1:
                _draw_trail(img, pts, colour, tlen)

            # bounding box at exactly this frame
            if frame_idx in data[tid]:
                d = data[tid][frame_idx]

                # Skip drawing box if interpolated (missing w/h)
                if np.isnan(d["w"]) or np.isnan(d["h"]):
                    continue

                _draw_box(img, d["cx"], d["cy"], d["w"], d["h"],
                        colour, tid, d["label"])

        # HUD
        hud = f"Frame {frame_idx}/{total_frames - 1}  Trail:{tlen}  {'PLAY' if playing else 'PAUSE'}"
        cv2.putText(img, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0),       1, cv2.LINE_AA)
        return img



   
    # ── event loop ────────────────────────────────────────────────────────────
    playing   = True
    cur_frame = start_frame
    frame_count = 0


    while True:
        img = _render(cur_frame, trail_len, playing)

        if img is not None:
            out.write(img)
        if img is None:
            break

        if frame_count > duration:
            break

        try:
            _, _, ww, wh = cv2.getWindowImageRect(win)
        except Exception:
            ww, wh = vid_w, vid_h
        if ww <= 0 or wh <= 0:
            ww, wh = vid_w, vid_h

        cv2.imshow(win, _fit(img, ww, wh))

        key = cv2.waitKey(delay_ms if playing else 30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            playing = not playing
        elif key in (81, ord('a')):                         # left / A
            playing   = False
            cur_frame = max(0, cur_frame - 1)
        elif key in (83, ord('d')):                         # right / D
            playing   = False
            cur_frame = min(total_frames - 1, cur_frame + 1)
        elif key in (ord('+'), ord('=')):
            trail_len = min(trail_len + 5, 500)
        elif key == ord('-'):
            trail_len = max(1, trail_len - 5)
        elif key == ord('s'):
            fname = f"frame_{cur_frame:06d}.png"
            cv2.imwrite(fname, img)
            print(f"[trace_viewer] saved {fname}")
        else:
            if playing:
                cur_frame += 1
                frame_count += 1
                if cur_frame >= total_frames:
                    playing   = False
                    cur_frame = total_frames - 1

        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# ── example usage (replace with your own variables) ───────────────────────────
if __name__ == "__main__":
    import pandas as pd

    df         = pd.read_csv("your_traces.csv", sep=None, engine="python")
    video_path = "your_video.mp4"

    view_traces(df, video_path, trail_len=30, scale=0.6)
