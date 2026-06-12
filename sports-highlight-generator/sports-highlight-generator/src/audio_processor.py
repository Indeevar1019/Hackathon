"""
audio_processor.py
1) Transcribe commentary with OpenAI Whisper (keyword-based excitement cues).
2) Detect crowd-noise / loudness spikes via short-time RMS energy.

Usage:
    python src/audio_processor.py --audio data/audio.wav --out results/audio.json
"""
import argparse
import json
import os

import numpy as np

EXCITEMENT_KEYWORDS = [
    # football
    "goal", "goooal", "scores", "what a strike", "penalty", "red card", "save",
    # cricket
    "six", "sixer", "four", "wicket", "out", "bowled", "caught", "century",
    # basketball
    "dunk", "slam", "three pointer", "buzzer",
    # generic
    "incredible", "unbelievable", "amazing", "brilliant", "fantastic",
    "what a", "oh my", "stunning", "magnificent", "sensational",
]


def transcribe(audio_path: str, model_size: str = "base", device: str = None):
    import torch
    import whisper
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[audio] Loading Whisper '{model_size}' on {device} ...")
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(audio_path, fp16=(device == "cuda"))
    segments = [
        {"start": round(s["start"], 2), "end": round(s["end"], 2),
         "text": s["text"].strip()}
        for s in result["segments"]
    ]
    print(f"[audio] Transcribed {len(segments)} segments")
    return segments


def keyword_score(text: str) -> float:
    t = text.lower()
    hits = sum(1 for k in EXCITEMENT_KEYWORDS if k in t)
    # exclamation marks and ALL CAPS also signal excitement
    hits += t.count("!") * 0.5
    return min(1.0, hits / 3.0)


def crowd_energy(audio_path: str, window_s: float = 1.0):
    """Per-second RMS loudness, normalized 0-1, with z-score spike flags."""
    import soundfile as sf
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    win = int(sr * window_s)
    n = len(data) // win
    rms = np.array([np.sqrt(np.mean(data[i*win:(i+1)*win] ** 2)) for i in range(n)])
    if rms.max() > 0:
        norm = rms / rms.max()
    else:
        norm = rms
    mu, sd = norm.mean(), norm.std() + 1e-8
    z = (norm - mu) / sd
    return [
        {"t": float(i * window_s), "energy": float(norm[i]),
         "spike": bool(z[i] > 1.5)}
        for i in range(n)
    ]


def run(audio_path: str, out_path: str = "results/audio.json",
        model_size: str = "base"):
    segments = transcribe(audio_path, model_size)
    for s in segments:
        s["excitement"] = round(keyword_score(s["text"]), 3)
    energy = crowd_energy(audio_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = {"segments": segments, "crowd_energy": energy}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[audio] Results -> {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default="results/audio.json")
    ap.add_argument("--model", default="base", help="Whisper size: tiny/base/small")
    args = ap.parse_args()
    run(args.audio, args.out, args.model)
