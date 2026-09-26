"""
team_assigner.py

Assigns each tracked player to one of 2 teams by clustering jersey color
with K-means, instead of the original zero-shot Fashion CLIP classifier.
No external model download, no internet dependency, and much cheaper per
frame — trades off some robustness to lighting/viewing-angle variation
versus CLIP.
"""

import sys
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans

sys.path.append("../")
from utils import read_stub, save_stub


class TeamAssigner:
    def __init__(self, n_teams: int = 2):
        self.n_teams = n_teams
        self.kmeans: Optional[KMeans] = None
        self.player_team_dict: Dict[int, int] = {}

    def _get_jersey_pixels(self, frame, bbox) -> Optional[np.ndarray]:
        """
        Crop the upper half of a player's bounding box (torso/jersey area,
        avoids shorts and the court background below the player's waist).
        """
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        torso = crop[: max(1, int(crop.shape[0] * 0.5)), :]
        # float64: sklearn's compiled KMeans lloyd_iter_chunked_dense expects
        # 'const double' buffers — float32 raises a dtype mismatch at predict().
        pixels = torso.reshape(-1, 3).astype(np.float64)
        return pixels if len(pixels) >= 2 else None

    def _dominant_jersey_color(self, pixels: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Splits the cropped torso region into 2 color clusters (jersey vs.
        background/skin) and returns the cluster center that is NOT the
        one dominating the crop's corners (a cheap background heuristic —
        corners of a tight player crop are usually background, not jersey).
        """
        if pixels is None:
            return None

        clustering = KMeans(n_clusters=2, n_init=3, random_state=42).fit(pixels)
        labels = clustering.labels_

        corner_labels = [labels[0], labels[-1]]
        bg_label = max(set(corner_labels), key=corner_labels.count)
        jersey_label = 1 - bg_label

        return clustering.cluster_centers_[jersey_label]

    def _fit_team_clusters(self, dominant_colors: np.ndarray) -> None:
        """
        Fits a top-level KMeans over all players' dominant jersey colors,
        splitting them into `n_teams` groups. Called once, on the first
        frame with enough players to calibrate against.
        """
        self.kmeans = KMeans(n_clusters=self.n_teams, n_init=10, random_state=42)
        self.kmeans.fit(dominant_colors)

    def get_player_teams_across_frames(
        self,
        video_frames: List,
        player_tracks: List[Dict[int, Dict[str, List[float]]]],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> List[Dict[int, int]]:
        """
        Assigns a team_id (1..n_teams) to every tracked player in every
        frame, with caching per stub file (identical structure to the
        original CLIP-based implementation).
        """
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None and len(player_assignment) == len(video_frames):
            return player_assignment

        # Step 1: calibrate team color clusters from the first frame that
        # has enough players detected to be representative of both teams.
        calibration_colors = []
        for frame_num, track in enumerate(player_tracks):
            if len(track) < 2:
                continue
            for _, data in track.items():
                color = self._dominant_jersey_color(self._get_jersey_pixels(video_frames[frame_num], data["bbox"]))
                if color is not None:
                    calibration_colors.append(color)
            if len(calibration_colors) >= self.n_teams:
                break

        if len(calibration_colors) >= self.n_teams:
            self._fit_team_clusters(np.array(calibration_colors))

        # Step 2: assign every player in every frame, caching by player_id
        # so a given track is only classified once (matches original
        # behaviour, avoids flicker between team 1/2 frame to frame).
        player_assignment = []
        for frame_num, track in enumerate(player_tracks):
            frame_assignment: Dict[int, int] = {}

            for player_id, data in track.items():
                if player_id in self.player_team_dict:
                    frame_assignment[player_id] = self.player_team_dict[player_id]
                    continue

                if self.kmeans is None:
                    frame_assignment[player_id] = 1
                    continue

                color = self._dominant_jersey_color(self._get_jersey_pixels(video_frames[frame_num], data["bbox"]))
                if color is None:
                    frame_assignment[player_id] = 1
                    continue

                team_id = int(self.kmeans.predict([color])[0]) + 1
                self.player_team_dict[player_id] = team_id
                frame_assignment[player_id] = team_id

            player_assignment.append(frame_assignment)

        if stub_path:
            save_stub(stub_path, player_assignment)

        return player_assignment
