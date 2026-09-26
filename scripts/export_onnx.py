"""
export_onnx.py

Exports the player detector and court keypoint (pose) models to ONNX,
saved right next to their .pt checkpoints (e.g. best.pt -> best.onnx) so
configs.MODEL_FORMAT="onnx" picks them up automatically (see
configs/configs.py's _resolve_model_path).

`dynamic=True` matters: without it, the export bakes in batch size 1, but
PlayerTracker/CourtKeypointDetector call .track()/.predict() with batches
of ~20 frames — a static-batch export raises an
`onnxruntime.InvalidArgument` shape mismatch the first time it's actually
used (see chat history / DISCOVERY_REPORT.md for the exact error).

Usage:
    python scripts/export_onnx.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

from configs.configs import _PLAYER_WEIGHT_BASE, _COURT_WEIGHT_BASE  # noqa: E402

IMGSZ = 640


def export(pt_path: str, label: str) -> str:
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"{label} weight not found: {pt_path}")

    onnx_path = os.path.splitext(pt_path)[0] + ".onnx"
    if os.path.exists(onnx_path):
        print(f"[{label}] {onnx_path} already exists, re-exporting to make sure it's current...")

    print(f"[{label}] exporting {pt_path} -> ONNX (imgsz={IMGSZ}, dynamic batch)...")
    model = YOLO(pt_path)
    exported_path = model.export(format="onnx", imgsz=IMGSZ, dynamic=True)
    print(f"[{label}] done: {exported_path}")
    return exported_path


def main():
    export(_PLAYER_WEIGHT_BASE, "player_detector")
    export(_COURT_WEIGHT_BASE, "court_keypoint_detector")
    print("\nSet MODEL_FORMAT=onnx in .env to use these instead of the .pt checkpoints.")


if __name__ == "__main__":
    main()
