import os

from dotenv import load_dotenv

load_dotenv()  # reads .env if present (gitignored — see .env.example)

STUBS_DEFAULT_PATH = 'stubs'

# Model weights — not committed to the repo (see README for how to obtain
# them). Override via PLAYER_WEIGHT_PATH / COURT_WEIGHT_PATH in .env;
# defaults to models/*.pt if unset.
#
# MODEL_FORMAT switches between the trained .pt checkpoint and its ONNX
# export (see scripts/export_onnx.py) without touching any pipeline code —
# Ultralytics' YOLO() picks the right backend from the file extension, so
# PlayerTracker/CourtKeypointDetector don't need to know which one is
# loaded. Falls back to .pt if the matching .onnx hasn't been exported yet,
# so setting MODEL_FORMAT=onnx before exporting doesn't break anything.
MODEL_FORMAT = os.getenv('MODEL_FORMAT', 'pt').strip().lower()


def _resolve_model_path(base_path: str, fmt: str) -> str:
    if fmt == 'onnx':
        onnx_path = os.path.splitext(base_path)[0] + '.onnx'
        if os.path.exists(onnx_path):
            return onnx_path
    return base_path


_PLAYER_WEIGHT_BASE = os.getenv('PLAYER_WEIGHT_PATH', 'models/player_detector.pt')
_COURT_WEIGHT_BASE = os.getenv('COURT_WEIGHT_PATH', 'models/court_keypoint_detector.pt')

PLAYER_DETECTOR_PATH = _resolve_model_path(_PLAYER_WEIGHT_BASE, MODEL_FORMAT)
COURT_KEYPOINT_DETECTOR_PATH = _resolve_model_path(_COURT_WEIGHT_BASE, MODEL_FORMAT)

# Human-readable name for run_info/API metadata — never expose the raw
# filesystem path there (reveals local directory structure for no benefit
# to the caller). See REPORT.md §3.2 for the real training numbers.
COURT_KEYPOINT_DETECTOR_NAME = 'YOLOv8m-pose'

# Player detector model registry — lets the API/UI offer a choice of
# architecture per run instead of always using the deployed default. Real
# mAP numbers behind each label come from the ablation in REPORT.md §3 —
# not placeholders. "yolov8m" keeps using the .env-configured path (same
# as PLAYER_DETECTOR_PATH above); the other two point at real weights
# uploaded under training/ablation/. MODEL_FORMAT (pt/onnx) still applies
# to whichever one is selected, via the same _resolve_model_path fallback.
DEFAULT_PLAYER_MODEL = 'yolov8m'

PLAYER_MODEL_REGISTRY = {
    'yolov8m': {
        'label': 'YOLOv8m',
        'description': 'Mặc định đang deploy · mAP50-95=0.527 · 25.8M tham số',
        'path_base': _PLAYER_WEIGHT_BASE,
    },
    'yolov8n': {
        'label': 'YOLOv8n',
        'description': 'Nhẹ & nhanh hơn · mAP50-95=0.356 · 3.0M tham số',
        'path_base': os.path.join('training', 'ablation', 'yolov8n', 'weights', 'best.pt'),
    },
    'rtdetr-l': {
        'label': 'RT-DETR-L',
        'description': 'mAP50-95 cao nhất (đỉnh 0.601) · 32.0M tham số · chưa benchmark latency',
        'path_base': os.path.join('training', 'ablation', 'rtdetr-l', 'weights', 'best.pt'),
    },
}


def get_player_detector_path(model_id: str = DEFAULT_PLAYER_MODEL) -> str:
    """Resolves a player_model id from PLAYER_MODEL_REGISTRY to an actual
    weight path, applying the same MODEL_FORMAT (pt/onnx) switch. Falls
    back to the default model if given an unknown id."""
    entry = PLAYER_MODEL_REGISTRY.get(model_id, PLAYER_MODEL_REGISTRY[DEFAULT_PLAYER_MODEL])
    return _resolve_model_path(entry['path_base'], MODEL_FORMAT)

# Ultralytics tracker config for the player tracker (BoT-SORT + camera motion
# compensation + Re-ID). See trackers/botsort.yaml.
PLAYER_TRACKER_CONFIG_PATH = os.path.join('trackers', 'botsort.yaml')

COURT_IMAGE_PATH = os.path.join('tactical_view', 'court_images', 'basketball_court.png')

# Each `python main.py` run gets its own output/run_N/ folder (video,
# heatmaps, run_info.json) via utils.create_run_dir() — see main.py.
OUTPUT_ROOT_DIR = 'output'

# Team colors (BGR, for cv2) — single source of truth shared by
# PlayerTracksDrawer, TacticalViewDrawer and HeatmapGenerator so a team's
# color is consistent between the annotated video and its heatmap.
TEAM_COLORS = {
    1: (255, 245, 238),
    2: (128, 0, 0),
}
