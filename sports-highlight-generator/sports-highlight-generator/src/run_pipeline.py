"""
run_pipeline.py — one-shot end-to-end runner.

Usage:
    python src/run_pipeline.py --input data/sample_video.mp4 --fps 1
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import frame_extractor
import audio_processor
import visual_analyzer
import event_detector
import highlight_generator
import caption_generator
import thumbnail_creator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--blip2", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("STEP 1/6: frames + audio")
    frame_extractor.extract_frames(args.input, "data/frames", args.fps)
    frame_extractor.extract_audio(args.input, "data/audio.wav")

    print("=" * 60)
    print("STEP 2/6: visual analysis (YOLOv8)")
    visual_analyzer.analyze_frames("data/frames", "results/visual.json")

    print("=" * 60)
    print("STEP 3/6: audio analysis (Whisper + crowd energy)")
    audio_processor.run("data/audio.wav", "results/audio.json", args.whisper)

    print("=" * 60)
    print("STEP 4/6: event fusion + detection")
    event_detector.run("results/visual.json", "results/audio.json",
                       "results/events.json")

    print("=" * 60)
    print("STEP 5/6: highlight clips")
    events = highlight_generator.run(args.input, "results/events.json")
    highlight_generator.make_reel(events)

    print("=" * 60)
    print("STEP 6/6: captions + thumbnails")
    caption_generator.run("results/events.json", "results/audio.json",
                          "data/frames", use_blip2=args.blip2)
    thumbnail_creator.run("results/events.json")

    with open("results/events.json") as f:
        payload = json.load(f)
    print("\nFINAL OUTPUT:")
    print(json.dumps(payload["events"], indent=2))


if __name__ == "__main__":
    main()
