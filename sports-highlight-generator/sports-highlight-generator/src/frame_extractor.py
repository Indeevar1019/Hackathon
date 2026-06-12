"""
frame_extractor.py
Extract frames (at a configurable FPS) and the audio track from a sports video.

Usage:
    python src/frame_extractor.py --input data/sample_video.mp4 --fps 1
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2


def extract_frames(video_path: str, out_dir: str = "data/frames", fps: float = 1.0):
    """Sample frames from the video every 1/fps seconds. Returns frame index metadata."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(native_fps / fps)))

    meta = []
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            ts = idx / native_fps
            fname = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            meta.append({"frame": fname, "timestamp": round(ts, 3)})
            saved += 1
        idx += 1
    cap.release()

    with open(os.path.join(out_dir, "frames_index.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[frame_extractor] Saved {saved} frames "
          f"(video fps={native_fps:.2f}, total frames={total}) -> {out_dir}")
    return meta


def extract_audio(video_path: str, out_path: str = "data/audio.wav", sr: int = 16000):
    """Extract mono 16 kHz WAV using ffmpeg (Whisper-friendly)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn",
           "-ac", "1", "-ar", str(sr), out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"[frame_extractor] Audio extracted -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--frames_dir", default="data/frames")
    ap.add_argument("--audio_out", default="data/audio.wav")
    args = ap.parse_args()

    extract_frames(args.input, args.frames_dir, args.fps)
    extract_audio(args.input, args.audio_out)
