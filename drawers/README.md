# 🖍️ Drawers

Modules for visualizing tracking + tactical-view results on video frames.
Ball/possession/pass drawers were removed (out of scope — see
`PRD_CLAUDE_CODE.md`).

---

## 📁 Contents

- `player_tracks_drawer.py`
  ➤ Draws **ellipses** under each tracked player, team-colored, with their
  **track ID** in a filled rectangle above their position.

- `court_keypoints_drawer.py`
  ➤ Visualizes detected court keypoints (index-labelled circles). Rewritten
  to consume the plain numpy array `CourtKeypointDetector` actually returns
  (no longer depends on `supervision`).

- `tactical_view_drawer.py`
  ➤ Overlays the top-down tactical court image on the frame, with
  team-colored dots for each player's transformed position. Ball-holder
  highlighting and hardcoded per-player-ID debug logic (from the previous
  scope) were removed.

- `frame_number_drawer.py`
  ➤ Draws the current frame number — useful for QA'ing tracker/keypoint
  output against the source video. Referenced by the original `main.py` but
  never implemented; added here.

---

## ✅ Example Usage

```python
from drawers import (
    PlayerTracksDrawer,
    CourtKeypointsDrawer,
    TacticalViewDrawer,
    FrameNumberDrawer,
)

player_drawer = PlayerTracksDrawer()
court_drawer = CourtKeypointsDrawer()
tactical_drawer = TacticalViewDrawer()
frame_number_drawer = FrameNumberDrawer()

frames = player_drawer.draw(video_frames, player_tracks, player_assignment)
frames = court_drawer.draw(frames, court_keypoints_per_frame)
frames = frame_number_drawer.draw(frames)
frames = tactical_drawer.draw(
    frames, court_image_path, width, height,
    key_points, tactical_player_positions, player_assignment,
)
```
