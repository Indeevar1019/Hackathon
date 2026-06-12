"""
visual_analyzer.py
Run YOLOv8 on sampled frames. Produces a per-frame "visual interest" score
based on detected objects (players, ball, goal area density, celebrations).

Usage:
    python src/visual_analyzer.py --frames_dir data/frames --out results/visual.json
"""
import argparse
import glob
import json
import os

# COCO classes of interest for sports
INTEREST_CLASSES = {
    "person": 0.15,        # many players clustered => something happening
    "sports ball": 0.6,    # ball visible & large => action near camera
    "baseball bat": 0.4,
    "tennis racket": 0.4,
    "frisbee": 0.3,
}


def analyze_frames(frames_dir: str, out_path: str = "results/visual.json",
                   model_name: str = "yolov8n.pt", conf: float = 0.35):
    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[visual] Loading {model_name} on device={device}")
    model = YOLO(model_name)

    index_path = os.path.join(frames_dir, "frames_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            frames = json.load(f)
    else:
        frames = [{"frame": p, "timestamp": i}
                  for i, p in enumerate(sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))))]

    results = []
    paths = [f["frame"] for f in frames]
    BATCH = 16
    for i in range(0, len(paths), BATCH):
        batch = paths[i:i + BATCH]
        preds = model(batch, conf=conf, device=device, verbose=False)
        for meta, pred in zip(frames[i:i + BATCH], preds):
            dets, score = [], 0.0
            persons = 0
            for box in pred.boxes:
                cls_name = model.names[int(box.cls)]
                c = float(box.conf)
                dets.append({"class": cls_name, "conf": round(c, 3)})
                if cls_name == "person":
                    persons += 1
                score += INTEREST_CLASSES.get(cls_name, 0.0) * c
            # player-density bonus: crowded penalty box / celebration huddle
            if persons >= 6:
                score += 0.3
            results.append({
                "frame": meta["frame"],
                "timestamp": meta["timestamp"],
                "detections": dets,
                "persons": persons,
                "visual_score": round(min(1.0, score), 3),
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[visual] Analyzed {len(results)} frames -> {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default="data/frames")
    ap.add_argument("--out", default="results/visual.json")
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()
    analyze_frames(args.frames_dir, args.out, args.model)
