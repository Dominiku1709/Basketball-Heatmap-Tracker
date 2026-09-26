# 🧢 Team Assigner

Assigns tracked players to one of 2 teams by clustering jersey color with
**K-means**, on the cropped torso region of each player's bounding box.

- `team_assigner.py` — Main class `TeamAssigner`. Calibrates a top-level
  2-cluster KMeans on the first frame with enough players detected, then
  classifies every subsequent player by nearest cluster, caching the result
  per `track_id`.

---

## 🚀 Features

- 🎨 **No external model / no internet** — replaces the original zero-shot
  Fashion CLIP classifier (which required downloading a ~600MB model from
  Hugging Face and ran one CLIP forward pass per player per frame). K-means
  is local, offline, and far cheaper per frame — the tradeoff is lower
  robustness to lighting/viewing-angle variation between jerseys of similar
  color.
- 🧠 Maintains player-team associations across frames (cached by `track_id`).
- 💾 Optional caching via `read_stub` / `save_stub` to skip recomputation.

---

## 🧱 Usage Example

```python
from team_assigner import TeamAssigner

assigner = TeamAssigner()
team_assignments = assigner.get_player_teams_across_frames(
    video_frames, player_tracks,
    read_from_stub=True, stub_path="stubs/player_assignment.pkl"
)
```
