# 🏆 Real-time Sports Highlight Generator (MULTIMODAL_003)
**AMD GPU Hackathon · Team: team-3195 · indeevar.ravinuthala**

Multi-modal AI pipeline that fuses **video frames (YOLOv8)**, **commentary audio (Whisper)**,
and **crowd-noise energy** to auto-detect key sports moments, then generates
**highlight clips, captions, and thumbnails** — all running on AMD GPUs via ROCm.

## Quick Start (on the AMD ROCm notebook)
```bash
pip install -r requirements.txt      # torch is pre-installed in the ROCm env
# put a public-dataset sports video at data/sample_video.mp4
python src/run_pipeline.py --input data/sample_video.mp4 --fps 1
```
Or open `notebooks/pipeline_demo.ipynb` → Run All.

## Pipeline
```
video ─► frame_extractor ─► YOLOv8 visual scores ─┐
      └► audio (ffmpeg) ─► Whisper excitement ────┼► fusion S(t)=0.45v+0.35c+0.20e
                         └► crowd RMS spikes ─────┘        │
                                              peak detection (adaptive threshold)
                                                           │
                              clips (ffmpeg) + captions (BLIP+commentary) + thumbnails
```

## Modules
| File | Role |
|---|---|
| `src/frame_extractor.py` | Sample frames @1 fps, extract 16 kHz mono WAV |
| `src/visual_analyzer.py` | YOLOv8 detection → per-frame visual-interest score |
| `src/audio_processor.py` | Whisper ASR + keyword excitement + crowd-energy spikes |
| `src/event_detector.py` | Multi-modal fusion + peak picking + event typing |
| `src/highlight_generator.py` | ffmpeg clip cutting + highlight reel concat |
| `src/caption_generator.py` | BLIP / BLIP-2 frame caption fused with commentary |
| `src/thumbnail_creator.py` | Sharpest action frame + event banner overlay |
| `src/run_pipeline.py` | End-to-end runner |

## Output (per event)
```json
{
  "event_id": 1,
  "timestamp_start": "00:12:34",
  "timestamp_end": "00:12:48",
  "event_type": "goal",
  "confidence": 0.92,
  "caption": "Goal around the 12' mark — a soccer player celebrating ...",
  "thumbnail": "output/thumbnails/event_1.jpg",
  "clip": "output/highlights/event_1.mp4"
}
```

## GPU-budget plan (4 h / 24 h)
Download models + video on CPU time first. GPU only for: YOLOv8 batch inference,
Whisper transcription, BLIP captioning. Use `yolov8n` + Whisper `base` for the first
end-to-end run; upgrade to `yolov8m` / `small` / BLIP-2 only if budget remains.

## Datasets (public only, per rules)
SoccerNet · SportsMOT · YouTube-8M sports · CC-licensed highlight clips.

## Learnings (fill in after runs)
- ROCm PyTorch exposes the standard `torch.cuda` API → zero code changes vs CUDA.
- ...

## Future Work
Real-time sliding-window streaming · vLLM auto-commentary · score-overlay OCR ·
vertical social clips with subtitles · sport-specific fine-tuned action models.
