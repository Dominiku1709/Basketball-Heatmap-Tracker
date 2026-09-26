# 🔥 Heatmap

New module — not in the original repo. Generates per-player and per-team
workrate heatmaps mapped onto real court (top-down) coordinates, per
`PRD_CLAUDE_CODE.md` mục 1.2 and mục 5.

- `heatmap_generator.py` — `HeatmapGenerator`. Takes the `tactical_view`
  module's already-computed court-space positions
  (`tactical_player_positions`, per `track_id`/frame) plus `player_assignment`
  (`track_id` → `team_id`), aggregates them across the whole clip, and
  renders a Gaussian-blurred density heatmap image per player and per team,
  blended over the court background image.

No model/weight is needed here — it's pure aggregation + rendering on top of
homography output that `tactical_view` already computes, so no perspective
transform is redone.

## Usage

```python
from heatmap import HeatmapGenerator

heatmap_generator = HeatmapGenerator(
    court_width=tactical_view_converter.width,
    court_height=tactical_view_converter.height,
    court_image_path=tactical_view_converter.court_image_path,
)
heatmap_generator.generate(tactical_player_positions, player_assignment, "output_heatmaps")
```

Output:
```
output_heatmaps/
├── players/
│   ├── player_3_heatmap.png
│   └── ...
└── teams/
    ├── team_1_heatmap.png
    └── team_2_heatmap.png
```
