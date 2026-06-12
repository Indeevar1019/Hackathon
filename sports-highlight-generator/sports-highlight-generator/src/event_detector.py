"""
event_detector.py
Fuse visual scores, commentary excitement, and crowd-noise spikes on a common
1-second timeline, then pick highlight windows via peak detection.

Fused score:  S(t) = 0.45*visual + 0.35*commentary + 0.20*crowd

Usage:
    python src/event_detector.py --visual_results results/visual.json \
                                 --audio_results results/audio.json \
                                 --out results/events.json
"""
import argparse
import json
import os

import numpy as np

W_VISUAL, W_COMMENTARY, W_CROWD = 0.45, 0.35, 0.20

EVENT_KEYWORD_MAP = {
    "goal": "goal", "scores": "goal", "penalty": "penalty",
    "wicket": "wicket", "bowled": "wicket", "caught": "wicket",
    "six": "boundary", "four": "boundary",
    "dunk": "dunk", "three pointer": "three_pointer",
    "save": "save", "red card": "card", "foul": "foul",
}


def build_timeline(visual, audio, duration=None):
    if duration is None:
        t_v = max((f["timestamp"] for f in visual), default=0)
        t_a = max((e["t"] for e in audio["crowd_energy"]), default=0)
        duration = int(max(t_v, t_a)) + 1
    T = int(duration)
    vis = np.zeros(T)
    com = np.zeros(T)
    crd = np.zeros(T)

    for f in visual:
        t = int(f["timestamp"])
        if t < T:
            vis[t] = max(vis[t], f["visual_score"])

    for s in audio["segments"]:
        for t in range(int(s["start"]), min(T, int(s["end"]) + 1)):
            com[t] = max(com[t], s.get("excitement", 0.0))

    for e in audio["crowd_energy"]:
        t = int(e["t"])
        if t < T:
            crd[t] = e["energy"] + (0.3 if e["spike"] else 0.0)
    crd = np.clip(crd, 0, 1)

    fused = W_VISUAL * vis + W_COMMENTARY * com + W_CROWD * crd
    # light smoothing
    kernel = np.array([0.25, 0.5, 0.25])
    fused = np.convolve(fused, kernel, mode="same")
    return fused, vis, com, crd


def classify_event(audio_segments, t_start, t_end):
    texts = " ".join(s["text"].lower() for s in audio_segments
                     if s["start"] <= t_end and s["end"] >= t_start)
    for kw, label in EVENT_KEYWORD_MAP.items():
        if kw in texts:
            return label
    return "key_moment"


def detect_events(fused, audio, threshold=None, pre=6, post=8,
                  min_gap=15, max_events=10):
    if threshold is None:
        threshold = float(fused.mean() + 1.2 * fused.std())
    peaks = []
    last = -min_gap
    order = np.argsort(fused)[::-1]
    for t in order:
        if fused[t] < threshold:
            break
        if all(abs(t - p) >= min_gap for p in peaks):
            peaks.append(int(t))
        if len(peaks) >= max_events:
            break
    peaks.sort()

    events = []
    T = len(fused)
    for i, p in enumerate(peaks, 1):
        start = max(0, p - pre)
        end = min(T - 1, p + post)
        events.append({
            "event_id": i,
            "peak_time": p,
            "start": start,
            "end": end,
            "timestamp_start": f"{start//3600:02d}:{(start//60)%60:02d}:{start%60:02d}",
            "timestamp_end": f"{end//3600:02d}:{(end//60)%60:02d}:{end%60:02d}",
            "confidence": round(float(min(1.0, fused[p])), 3),
            "event_type": classify_event(audio["segments"], start, end),
        })
    return events, threshold


def run(visual_path, audio_path, out_path="results/events.json", **kw):
    with open(visual_path) as f:
        visual = json.load(f)
    with open(audio_path) as f:
        audio = json.load(f)

    fused, vis, com, crd = build_timeline(visual, audio)
    events, thr = detect_events(fused, audio, **kw)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "threshold": round(thr, 4),
        "events": events,
        "fused_scores": [round(float(x), 4) for x in fused],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[events] {len(events)} events detected (thr={thr:.3f}) -> {out_path}")
    for e in events:
        print(f"   #{e['event_id']} {e['event_type']:>13} "
              f"{e['timestamp_start']}–{e['timestamp_end']} conf={e['confidence']}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual_results", default="results/visual.json")
    ap.add_argument("--audio_results", default="results/audio.json")
    ap.add_argument("--out", default="results/events.json")
    args = ap.parse_args()
    run(args.visual_results, args.audio_results, args.out)
