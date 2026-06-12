"""
highlight_generator.py
Cut highlight clips from the source video for each detected event using ffmpeg
(stream copy when possible — fast, no GPU needed).

Usage:
    python src/highlight_generator.py --video data/sample_video.mp4 \
                                      --events results/events.json
"""
import argparse
import json
import os
import subprocess


def cut_clip(video, start, end, out_path):
    duration = max(1, end - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", video, "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def run(video, events_path, out_dir="output/highlights"):
    os.makedirs(out_dir, exist_ok=True)
    with open(events_path) as f:
        events = json.load(f)["events"]

    for e in events:
        out_path = os.path.join(out_dir, f"event_{e['event_id']}.mp4")
        cut_clip(video, e["start"], e["end"], out_path)
        e["clip"] = out_path
        print(f"[highlights] event {e['event_id']} -> {out_path}")

    # write back enriched events
    with open(events_path) as f:
        payload = json.load(f)
    payload["events"] = events
    with open(events_path, "w") as f:
        json.dump(payload, f, indent=2)
    return events


def make_reel(events, out_path="output/highlights/highlight_reel.mp4"):
    """Concatenate all event clips into one reel."""
    if not events:
        return None
    list_file = "output/highlights/concat.txt"
    with open(list_file, "w") as f:
        for e in events:
            f.write(f"file '{os.path.abspath(e['clip'])}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", list_file, "-c", "copy", out_path],
                   check=True, capture_output=True)
    print(f"[highlights] reel -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--events", default="results/events.json")
    ap.add_argument("--out_dir", default="output/highlights")
    ap.add_argument("--reel", action="store_true")
    args = ap.parse_args()
    evs = run(args.video, args.events, args.out_dir)
    if args.reel:
        make_reel(evs)
