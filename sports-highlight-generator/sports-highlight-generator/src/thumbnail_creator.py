"""
thumbnail_creator.py
Pick the sharpest, most action-dense frame inside each event window and render
a thumbnail with the event-type banner overlaid.

Usage:
    python src/thumbnail_creator.py --events results/events.json --frames_dir data/frames
"""
import argparse
import json
import os

import cv2
import numpy as np


def sharpness(img) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def pick_best_frame(frames, visual, start, end):
    """Best = high visual_score, tie-broken by sharpness."""
    cand = [f for f in frames if start <= f["timestamp"] <= end]
    if not cand:
        cand = frames
    vmap = {v["frame"]: v.get("visual_score", 0) for v in visual}
    cand.sort(key=lambda f: vmap.get(f["frame"], 0), reverse=True)
    top = cand[:5]
    best, best_s = top[0]["frame"], -1
    for f in top:
        img = cv2.imread(f["frame"])
        if img is None:
            continue
        s = sharpness(img)
        if s > best_s:
            best, best_s = f["frame"], s
    return best


def render(src_path, event, out_path, width=1280):
    img = cv2.imread(src_path)
    h, w = img.shape[:2]
    scale = width / w
    img = cv2.resize(img, (width, int(h * scale)))
    h, w = img.shape[:2]

    # banner
    banner_h = 70
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    label = f"{event['event_type'].replace('_', ' ').upper()}  |  " \
            f"{event['timestamp_start']}  |  conf {event['confidence']:.2f}"
    cv2.putText(img, label, (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def run(events_path, frames_dir="data/frames",
        visual_path="results/visual.json", out_dir="output/thumbnails"):
    os.makedirs(out_dir, exist_ok=True)
    with open(events_path) as f:
        payload = json.load(f)
    with open(os.path.join(frames_dir, "frames_index.json")) as f:
        frames = json.load(f)
    visual = []
    if os.path.exists(visual_path):
        with open(visual_path) as f:
            visual = json.load(f)

    for e in payload["events"]:
        src = pick_best_frame(frames, visual, e["start"], e["end"])
        out_path = os.path.join(out_dir, f"event_{e['event_id']}.jpg")
        render(src, e, out_path)
        e["thumbnail"] = out_path
        print(f"[thumb] #{e['event_id']} -> {out_path}")

    with open(events_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload["events"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="results/events.json")
    ap.add_argument("--frames_dir", default="data/frames")
    ap.add_argument("--visual", default="results/visual.json")
    args = ap.parse_args()
    run(args.events, args.frames_dir, args.visual)
