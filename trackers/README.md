# 🛰️ Trackers

Player detection + tracking for video analysis. Ball tracking was removed
(out of scope — see `PRD_CLAUDE_CODE.md`).

---

## 📄 Files

- 🎽 `player_tracker.py` – Player detection (YOLO) + tracking via Ultralytics'
  built-in **BoT-SORT** tracker, configured in `botsort.yaml` with camera
  motion compensation (`gmc_method: sparseOptFlow`) and Re-ID
  (`with_reid: true`). This replaces the original plain
  `supervision.ByteTrack()` setup, which had no camera-motion handling and
  fragmented player IDs under continuous broadcast-style camera pan.
- ⚙️ `botsort.yaml` – Tracker config, tune here (thresholds, buffer, GMC method).
- 📦 `__init__.py` – Marks the folder as a Python package.

---

## 🚀 Usage Example

```python
from trackers import PlayerTracker
from utils.video_utils import read_video

tracker = PlayerTracker("models/player_detector.pt")

frames = read_video("input_videos/video_1.mp4")

tracks = tracker.get_object_tracks(
    frames,
    read_from_stub=True,
    stub_path="stubs/player_tracks.pkl"
)
```
