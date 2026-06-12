"""
caption_generator.py
Generate a natural-language caption per highlight by combining:
  1) BLIP image captioning of the peak frame  (what we SEE)
  2) Whisper commentary in the event window   (what we HEAR)

Falls back to Salesforce/blip-image-captioning-base if BLIP-2 is too heavy
for the remaining GPU budget.

Usage:
    python src/caption_generator.py --events results/events.json \
                                    --audio results/audio.json \
                                    --frames_dir data/frames
"""
import argparse
import glob
import json
import os

from PIL import Image

_model = None
_processor = None
_device = None


def load_model(use_blip2: bool = False):
    global _model, _processor, _device
    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if _device == "cuda" else torch.float32
    if use_blip2:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        name = "Salesforce/blip2-opt-2.7b"
        _processor = Blip2Processor.from_pretrained(name)
        _model = Blip2ForConditionalGeneration.from_pretrained(
            name, torch_dtype=dtype).to(_device)
    else:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        name = "Salesforce/blip-image-captioning-base"
        _processor = BlipProcessor.from_pretrained(name)
        _model = BlipForConditionalGeneration.from_pretrained(
            name, torch_dtype=dtype).to(_device)
    print(f"[caption] Loaded {name} on {_device}")


def caption_image(img_path: str) -> str:
    import torch
    img = Image.open(img_path).convert("RGB")
    inputs = _processor(images=img, return_tensors="pt").to(
        _device, dtype=_model.dtype)
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=40)
    return _processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def nearest_frame(frames_dir: str, t: float) -> str:
    idx_path = os.path.join(frames_dir, "frames_index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            frames = json.load(f)
        return min(frames, key=lambda fr: abs(fr["timestamp"] - t))["frame"]
    files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    return files[min(int(t), len(files) - 1)]


def commentary_in_window(audio, start, end):
    return " ".join(s["text"] for s in audio["segments"]
                    if s["start"] <= end and s["end"] >= start).strip()


def compose(event, visual_caption, commentary):
    minute = event["peak_time"] // 60
    etype = event["event_type"].replace("_", " ").title()
    parts = [f"{etype} around the {minute}' mark"]
    if visual_caption:
        parts.append(f"— {visual_caption}")
    if commentary:
        parts.append(f'Commentary: "{commentary[:140]}"')
    return " ".join(parts)


def run(events_path, audio_path, frames_dir="data/frames",
        out_dir="output/captions", use_blip2=False):
    os.makedirs(out_dir, exist_ok=True)
    with open(events_path) as f:
        payload = json.load(f)
    with open(audio_path) as f:
        audio = json.load(f)

    load_model(use_blip2)
    for e in payload["events"]:
        frame = nearest_frame(frames_dir, e["peak_time"])
        vis_cap = caption_image(frame)
        comm = commentary_in_window(audio, e["start"], e["end"])
        e["caption"] = compose(e, vis_cap, comm)
        e["visual_caption"] = vis_cap
        with open(os.path.join(out_dir, f"event_{e['event_id']}.json"), "w") as f:
            json.dump(e, f, indent=2)
        print(f"[caption] #{e['event_id']}: {e['caption']}")

    with open(events_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload["events"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="results/events.json")
    ap.add_argument("--audio", default="results/audio.json")
    ap.add_argument("--frames_dir", default="data/frames")
    ap.add_argument("--blip2", action="store_true", help="use BLIP-2 (heavier)")
    args = ap.parse_args()
    run(args.events, args.audio, args.frames_dir, use_blip2=args.blip2)
