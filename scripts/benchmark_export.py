"""
benchmark_export.py

Real latency/FPS comparison: PyTorch (.pt) vs ONNXRuntime (.onnx), for both
the player detector and the court keypoint model, on real frames from an
input video. Answers yeucau.docx mục 7's "so tốc độ trước/sau export"
requirement with actual numbers instead of a guess.

CPU-only, single-threaded like the rest of this pipeline (no GPU available
in this dev environment) — that's the realistic deployment target for the
FastAPI demo (api/app.py), not a claim about GPU behaviour.

Usage:
    python scripts/benchmark_export.py [--video input_videos/video_1.mp4] [--frames 60] [--warmup 5]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ultralytics import YOLO

from configs.configs import _PLAYER_WEIGHT_BASE, _COURT_WEIGHT_BASE  # noqa: E402


def load_frames(video_path: str, n: int):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    return frames


def benchmark_model(weight_path: str, frames, warmup: int, task: str):
    model = YOLO(weight_path)

    predict_kwargs = {"conf": 0.5 if task == "detect" else 0.6, "verbose": False}

    # Warm-up: first-call overhead (session/graph init, memory allocation)
    # isn't representative of steady-state per-frame latency.
    for frame in frames[:warmup]:
        model.predict([frame], **predict_kwargs)

    per_frame_ms = []
    for frame in frames:
        t0 = time.perf_counter()
        model.predict([frame], **predict_kwargs)
        per_frame_ms.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(per_frame_ms) / len(per_frame_ms)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return {
        "avg_latency_ms": round(avg_ms, 2),
        "fps": round(fps, 2),
        "min_latency_ms": round(min(per_frame_ms), 2),
        "max_latency_ms": round(max(per_frame_ms), 2),
        "n_frames": len(frames),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNXRuntime inference latency")
    parser.add_argument("--video", default="input_videos/video_1.mp4")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading {args.frames} frames from {args.video}...")
    frames = load_frames(args.video, args.frames)
    print(f"Loaded {len(frames)} frames.\n")

    results = {}
    for label, pt_path, task in [
        ("player_detector", _PLAYER_WEIGHT_BASE, "detect"),
        ("court_keypoint_detector", _COURT_WEIGHT_BASE, "pose"),
    ]:
        onnx_path = os.path.splitext(pt_path)[0] + ".onnx"
        results[label] = {}

        print(f"=== {label} ===")
        print(f"  PyTorch ({pt_path})...")
        results[label]["pytorch"] = benchmark_model(pt_path, frames, args.warmup, task)
        print(f"    {results[label]['pytorch']}")

        if os.path.exists(onnx_path):
            print(f"  ONNXRuntime ({onnx_path})...")
            results[label]["onnxruntime"] = benchmark_model(onnx_path, frames, args.warmup, task)
            print(f"    {results[label]['onnxruntime']}")

            pt_ms = results[label]["pytorch"]["avg_latency_ms"]
            onnx_ms = results[label]["onnxruntime"]["avg_latency_ms"]
            speedup = pt_ms / onnx_ms if onnx_ms > 0 else float("nan")
            results[label]["onnx_speedup_x"] = round(speedup, 2)
            print(f"    speedup: {speedup:.2f}x {'(ONNX faster)' if speedup > 1 else '(PyTorch faster)'}")
        else:
            print(f"  No ONNX export found at {onnx_path} — run scripts/export_onnx.py first.")
        print()

    out_path = os.path.join("output", "benchmark_export.json")
    os.makedirs("output", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
