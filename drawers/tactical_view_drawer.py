# tactical_view_drawer.py

import cv2
import numpy as np

from configs import TEAM_COLORS


class TacticalViewDrawer:
    """
    Overlays a top-down tactical court image on each video frame, with
    team-colored dots for each player's transformed position.

    Ball-possession highlighting and hardcoded per-player-ID special cases
    (leftover debug code tied to one specific test video) were removed —
    ball tracking is out of scope for this build (see PRD_CLAUDE_CODE.md).

    `tactical_court_keypoints` is `TacticalViewConverter.key_points` — a
    fixed list of reference court-landmark positions in tactical (top-down)
    pixel space, constant across the whole clip (NOT one entry per video
    frame). The original repo's version of this file treated it as if it
    varied per frame (indexing `[idx]` into it and smoothing between
    frames), which throws given a flat list of (x, y) tuples — that code
    never actually ran before (see DISCOVERY_REPORT.md). Draw it as the
    static reference grid it is.
    """

    def __init__(self, team_1_color=TEAM_COLORS[1], team_2_color=TEAM_COLORS[2]):
        self.start_x = 20
        self.start_y = 40
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

        self.last_valid_positions = None
        self.last_valid_teams = None
        self.previous_smoothed_positions = {}

        self._court_img_cache_key = None
        self._court_img_cache = None

    def _interpolate_missing_positions(self, tactical_player_positions):
        """
        Linearly interpolate missing player positions frame-by-frame.

        Operates only on lightweight (x, y) coordinate dicts, not video
        frames — cheap to run once up front regardless of video length.
        """
        total_frames = len(tactical_player_positions)
        player_tracks = {}

        for f_idx, frame_dict in enumerate(tactical_player_positions):
            if not frame_dict:
                continue
            for pid, pos in frame_dict.items():
                if pid not in player_tracks:
                    player_tracks[pid] = {}
                player_tracks[pid][f_idx] = np.array(pos)

        interpolated_positions = [{} for _ in range(total_frames)]

        for pid, frames in player_tracks.items():
            for i in range(total_frames):
                if i in frames:
                    interpolated_positions[i][pid] = frames[i]
                else:
                    prev = next((j for j in reversed(range(0, i)) if j in frames), None)
                    next_ = next((j for j in range(i + 1, total_frames) if j in frames), None)

                    if prev is not None and next_ is not None:
                        alpha = (i - prev) / (next_ - prev)
                        interp_pos = (1 - alpha) * frames[prev] + alpha * frames[next_]
                        interpolated_positions[i][pid] = interp_pos
                    elif prev is not None:
                        interpolated_positions[i][pid] = frames[prev]
                    elif next_ is not None:
                        interpolated_positions[i][pid] = frames[next_]

        return interpolated_positions

    def prepare_positions(self, tactical_player_positions):
        """
        Interpolates missing positions once, up front. Call this before a
        streaming `draw_frame` loop and pass its result in as
        `interpolated_positions` — interpolation needs to look ahead to
        future frames, so it can't be done per-frame inside the streaming
        loop itself, but it's cheap (coordinates only) regardless of video
        length.
        """
        if not tactical_player_positions:
            return tactical_player_positions
        return self._interpolate_missing_positions(tactical_player_positions)

    def _get_court_image(self, court_image_path, width, height):
        cache_key = (court_image_path, width, height)
        if self._court_img_cache_key != cache_key:
            court_img = cv2.imread(court_image_path)
            if court_img is None:
                raise ValueError("Court image could not be loaded.")
            self._court_img_cache = cv2.resize(court_img, (width, height))
            self._court_img_cache_key = cache_key
        return self._court_img_cache

    def draw_frame(
        self,
        frame,
        idx,
        court_image_path,
        width,
        height,
        tactical_court_keypoints,
        interpolated_positions=None,
        player_assignment=None,
    ):
        """
        Draws the tactical overlay on one frame. `interpolated_positions`
        must already be the output of `prepare_positions()` — this method
        does not (and cannot, looking only at frame `idx`) interpolate.
        Mutates and returns `frame` in place.
        """
        court_img = self._get_court_image(court_image_path, width, height)
        frame[self.start_y:self.start_y + height, self.start_x:self.start_x + width] = court_img.copy()

        # Draw the fixed court-landmark reference grid (same every frame)
        if tactical_court_keypoints is not None:
            for x, y in tactical_court_keypoints:
                point = (int(x) + self.start_x, int(y) + self.start_y)
                cv2.circle(frame, point, 4, (0, 165, 255), -1)

        smoothed_positions = None
        if interpolated_positions and idx < len(interpolated_positions):
            current_positions = interpolated_positions[idx]

            smoothed_positions = {}
            alpha = 0.8
            new_previous = {}

            for pid, raw_pos in current_positions.items():
                curr_pos = np.array(raw_pos)
                if pid in self.previous_smoothed_positions:
                    prev_pos = self.previous_smoothed_positions[pid]
                    smooth_pos = alpha * prev_pos + (1 - alpha) * curr_pos
                else:
                    smooth_pos = curr_pos

                smoothed_positions[pid] = smooth_pos
                new_previous[pid] = smooth_pos

            self.previous_smoothed_positions = new_previous
            self.last_valid_positions = smoothed_positions
        else:
            smoothed_positions = self.last_valid_positions

        if player_assignment and idx < len(player_assignment) and player_assignment[idx]:
            self.last_valid_teams = player_assignment[idx]

        # Draw players present this frame
        if smoothed_positions and self.last_valid_teams:
            for pid, pos in smoothed_positions.items():
                team = self.last_valid_teams.get(pid, 1)
                if team == 1:
                    color = self.team_1_color
                    text_color = (0, 0, 0)  # Black text
                else:
                    color = self.team_2_color
                    text_color = (255, 255, 255)  # White text

                x, y = int(pos[0]) + self.start_x, int(pos[1]) + self.start_y
                cv2.circle(frame, (x, y), 8, color, -1)
                cv2.putText(frame, str(pid), (x - 5, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        return frame

    def draw(self, video_frames, court_image_path, width, height, tactical_court_keypoints,
             tactical_player_positions=None, player_assignment=None):
        """
        Draws the tactical overlay across a whole list of frames, returning
        a new list. For long videos prefer `prepare_positions()` +
        `draw_frame` in a streaming loop (see main.py).
        """
        interpolated_positions = self.prepare_positions(tactical_player_positions)
        return [
            self.draw_frame(
                frame.copy(), idx, court_image_path, width, height,
                tactical_court_keypoints, interpolated_positions, player_assignment,
            )
            for idx, frame in enumerate(video_frames)
        ]
