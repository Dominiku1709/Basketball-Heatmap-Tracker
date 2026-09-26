# player_tracks_drawer.py

"""
Draws tracked player positions on video frames using ellipses + track ID.
"""

from typing import Any, Dict, List

from configs import TEAM_COLORS

from .utils import draw_ellipse


class PlayerTracksDrawer:
    """
    Class for drawing tracked player positions on video frames using ellipses.
    """

    def __init__(self, team1_color=TEAM_COLORS[1], team2_color=TEAM_COLORS[2]):
        """
        Initializes the drawer with team colors.
        """
        self.default_player_team_id = 1
        self.team1_color = team1_color
        self.team2_color = team2_color

    def draw_frame(
        self,
        frame: Any,
        frame_tracks: Dict[int, Dict[str, Any]],
        frame_assignment: Dict[int, int],
    ) -> Any:
        """
        Draws ellipses + track ID for one frame's tracked players. Mutates
        and returns `frame` in place (no copy) — the caller owns whether
        the input frame needs preserving.
        """
        for track_id, player_data in frame_tracks.items():
            team_id = frame_assignment.get(track_id, self.default_player_team_id)
            color = self.team1_color if team_id == 1 else self.team2_color
            bbox = player_data.get('bbox')

            if bbox:
                frame = draw_ellipse(frame, bbox, color=color, track_id=track_id)

        return frame

    def draw(
        self,
        video_frames: List[Any],
        tracks: List[Dict[int, Dict[str, Any]]],
        player_assignment: List[Dict[int, int]],
    ) -> List[Any]:
        """
        Draws ellipses + track ID across a whole list of frames, returning a
        new list. Convenient for small clips; for long videos prefer calling
        `draw_frame` in a streaming loop (see main.py) so the whole output
        video isn't held in memory as a second full copy.
        """
        return [
            self.draw_frame(frame.copy(), tracks[frame_num], player_assignment[frame_num])
            for frame_num, frame in enumerate(video_frames)
        ]
