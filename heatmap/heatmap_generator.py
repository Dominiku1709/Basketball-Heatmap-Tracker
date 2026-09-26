"""
heatmap_generator.py

New module (not in the original repo — see PRD_CLAUDE_CODE.md mục 5/6).

Aggregates player positions in tactical (real court, top-down) coordinates
across an entire clip and renders workrate heatmaps per player and per
team. Deliberately reuses `TacticalViewConverter`'s output
(`tactical_player_positions`, already in court-space pixels via homography)
instead of redoing any perspective transform — this module is pure
aggregation + rendering, no model/weight required.

Each team gets its own heatmap color (from configs.TEAM_COLORS by default)
and every player's heatmap is tinted in their own team's color, so a
player's heatmap and their team's heatmap are visually consistent with each
other and with the team colors drawn on the annotated video.

Per-player heatmaps are a 2-panel matplotlib figure — the heatmap on the
left, a crop of that player (from whichever frame their detected bbox was
largest, as a cheap proxy for "clearest, least motion-blurred") on the
right — since we have no separate player-photo source, only bboxes on
video frames. Team heatmaps stay plain (no single "photo" makes sense for
a team of 5).
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless — no display backend needed/available on a server
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_TEAM_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (255, 245, 238),
    2: (128, 0, 0),
}
FALLBACK_COLOR: Tuple[int, int, int] = (200, 200, 200)  # unassigned/unknown team


class HeatmapGenerator:
    def __init__(
        self,
        court_width: int,
        court_height: int,
        court_image_path: Optional[str] = None,
        team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ):
        """
        `court_width`/`court_height` must match the tactical view's pixel
        dimensions (TacticalViewConverter.width/height) so accumulated
        positions line up with the court background image.
        """
        self.court_width = court_width
        self.court_height = court_height
        self.court_image_path = court_image_path
        self.team_colors = team_colors or DEFAULT_TEAM_COLORS

    def accumulate_positions(
        self,
        tactical_player_positions: List[Dict[int, List[float]]],
        player_assignment: List[Dict[int, int]],
    ) -> Tuple[
        Dict[int, List[Tuple[float, float]]],
        Dict[int, List[Tuple[float, float]]],
        Dict[int, int],
    ]:
        """
        Builds per-player and per-team position histories from the tactical
        view's frame-by-frame output, plus each player's team_id (most
        recent assignment seen — team_assigner caches per player_id, so this
        is stable across the clip in practice).

        Returns:
            player_positions: {track_id: [(x, y), ...]} over the whole clip
            team_positions: {team_id: [(x, y), ...]} over the whole clip
            player_team: {track_id: team_id}
        """
        player_positions: Dict[int, List[Tuple[float, float]]] = {}
        team_positions: Dict[int, List[Tuple[float, float]]] = {}
        player_team: Dict[int, int] = {}

        for frame_positions, frame_assignment in zip(tactical_player_positions, player_assignment):
            for player_id, pos in frame_positions.items():
                x, y = pos
                player_positions.setdefault(player_id, []).append((x, y))

                team_id = frame_assignment.get(player_id)
                if team_id is not None:
                    team_positions.setdefault(team_id, []).append((x, y))
                    player_team[player_id] = team_id

        return player_positions, team_positions, player_team

    def _render_heatmap_array(
        self,
        positions: List[Tuple[float, float]],
        color: Tuple[int, int, int],
        sigma: int = 15,
    ) -> np.ndarray:
        """
        Renders a density heatmap from a list of (x, y) court-space points
        into a BGR uint8 array, tinted in `color` instead of a fixed
        rainbow colormap — density fades from black (unvisited) to full
        `color` (most visited), so team/player identity is visible at a
        glance. Blended over the court background if available.
        """
        density = np.zeros((self.court_height, self.court_width), dtype=np.float32)

        for x, y in positions:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < self.court_width and 0 <= yi < self.court_height:
                density[yi, xi] += 1

        density = cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if density.max() > 0:
            density = density / density.max()

        color_layer = np.zeros((self.court_height, self.court_width, 3), dtype=np.float32)
        for channel in range(3):
            color_layer[:, :, channel] = density * color[channel]
        color_layer = color_layer.astype(np.uint8)

        if self.court_image_path and os.path.exists(self.court_image_path):
            court_img = cv2.imread(self.court_image_path)
            court_img = cv2.resize(court_img, (self.court_width, self.court_height))
            return cv2.addWeighted(court_img, 0.5, color_layer, 0.7, 0)
        return color_layer

    def _render_heatmap(
        self,
        positions: List[Tuple[float, float]],
        output_path: str,
        color: Tuple[int, int, int],
        sigma: int = 15,
    ) -> None:
        """Renders + writes a plain heatmap image (used for team heatmaps)."""
        output_img = self._render_heatmap_array(positions, color, sigma)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_img)

    @staticmethod
    def _find_best_player_crop(
        player_id: int,
        player_tracks: List[Dict[int, Dict[str, List[float]]]],
        video_frames: List,
    ) -> Optional[np.ndarray]:
        """
        Picks the frame where `player_id`'s detected bbox has the largest
        area (cheap proxy for "closest to camera, clearest, least
        motion-blurred") and returns that crop as a BGR array, or None if
        the player never appears in `player_tracks`.
        """
        best_area = -1.0
        best_bbox = None
        best_frame_idx = None

        for frame_idx, frame_tracks in enumerate(player_tracks):
            data = frame_tracks.get(player_id)
            if data is None:
                continue
            x1, y1, x2, y2 = data["bbox"]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best_bbox = (x1, y1, x2, y2)
                best_frame_idx = frame_idx

        if best_frame_idx is None or video_frames[best_frame_idx] is None:
            return None

        frame = video_frames[best_frame_idx]
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = best_bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _render_player_composite(
        self,
        heatmap_bgr: np.ndarray,
        player_crop_bgr: Optional[np.ndarray],
        player_id: int,
        team_id: Optional[int],
        output_path: str,
    ) -> None:
        """
        Saves a 2-panel figure: the heatmap on the left, a snapshot crop of
        that player on the right.
        """
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))

        axes[0].imshow(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Workrate Heatmap", fontsize=10)
        axes[0].axis("off")

        if player_crop_bgr is not None:
            axes[1].imshow(cv2.cvtColor(player_crop_bgr, cv2.COLOR_BGR2RGB))
        else:
            axes[1].text(0.5, 0.5, "No snapshot available", ha="center", va="center", fontsize=9)
        axes[1].set_title("Player Snapshot", fontsize=10)
        axes[1].axis("off")

        team_label = f"Team {team_id}" if team_id is not None else "Team unknown"
        fig.suptitle(f"Player #{player_id} — {team_label}", fontsize=12, fontweight="bold")
        fig.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=120)
        plt.close(fig)

    def generate(
        self,
        tactical_player_positions: List[Dict[int, List[float]]],
        player_assignment: List[Dict[int, int]],
        player_tracks: List[Dict[int, Dict[str, List[float]]]],
        video_frames: List,
        output_dir: str,
    ) -> List[Dict]:
        """
        Generates and saves one whole-clip heatmap per player (composited
        with a player snapshot, see `_render_player_composite`) and per
        team, under `output_dir/players/` and `output_dir/teams/`.

        `player_tracks`/`video_frames` must be the SAME frame-indexed data
        the rest of the pipeline used (e.g. main.py's `on_court_player_tracks`
        and the raw `video_frames`) — called before any frame gets freed by
        the streaming draw loop (see pipeline.py).

        Returns:
            list of {"player_id": int, "path": str} for every player
            heatmap written (used by the API to expose a gallery of them).
        """
        player_positions, team_positions, player_team = self.accumulate_positions(
            tactical_player_positions, player_assignment
        )

        player_heatmaps: List[Dict] = []
        for player_id, positions in player_positions.items():
            team_id = player_team.get(player_id)
            color = self.team_colors.get(team_id, FALLBACK_COLOR)
            heatmap_arr = self._render_heatmap_array(positions, color)
            crop = self._find_best_player_crop(player_id, player_tracks, video_frames)
            out_path = os.path.join(output_dir, "players", f"player_{player_id}_heatmap.png")
            self._render_player_composite(heatmap_arr, crop, player_id, team_id, out_path)
            player_heatmaps.append({"player_id": player_id, "path": out_path})

        for team_id, positions in team_positions.items():
            color = self.team_colors.get(team_id, FALLBACK_COLOR)
            out_path = os.path.join(output_dir, "teams", f"team_{team_id}_heatmap.png")
            self._render_heatmap(positions, out_path, color)

        return player_heatmaps
