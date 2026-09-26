"""
pipeline.py

The actual analysis pipeline (detect -> track -> court keypoints -> team
assignment -> tactical view -> heatmaps -> annotated video), extracted out
of main.py so both the CLI (main.py) and the FastAPI service (api/app.py)
call the same code instead of duplicating it.

Model instances are created fresh on every call (matches the original
CLI behaviour exactly) rather than reused across calls — Ultralytics'
BoT-SORT tracker keeps internal state tied to the model instance via
`persist=True`, and reusing one PlayerTracker across unrelated videos in a
server risks leaking track state from one video's tracking session into
the next. Reloading the ~50MB weight file per call costs a fraction of a
second, negligible next to the seconds-to-minutes of detection/tracking
inference itself — not worth the correctness risk to optimize away here.
"""

import os
import json
from datetime import datetime

from utils import read_video, get_video_fps, StreamingVideoWriter, create_run_dir
from trackers import PlayerTracker
from team_assigner import TeamAssigner
from Court_keypoint_detection import CourtKeypointDetector
from tactical_view import TacticalViewConverter
from heatmap import HeatmapGenerator
from drawers import (
    PlayerTracksDrawer,
    CourtKeypointsDrawer,
    FrameNumberDrawer,
    TacticalViewDrawer,
)
from configs import (
    STUBS_DEFAULT_PATH,
    PLAYER_TRACKER_CONFIG_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_NAME,
    COURT_IMAGE_PATH,
    OUTPUT_ROOT_DIR,
    TEAM_COLORS,
    MODEL_FORMAT,
    DEFAULT_PLAYER_MODEL,
    PLAYER_MODEL_REGISTRY,
    get_player_detector_path,
)


def run_analysis(
    input_video_path: str,
    output_root: str = OUTPUT_ROOT_DIR,
    stub_root: str = STUBS_DEFAULT_PATH,
    player_model: str = DEFAULT_PLAYER_MODEL,
) -> dict:
    """
    Runs the full pipeline on one video and returns the same dict that gets
    written to run_info.json: run_id, timestamp, frame/player counts, model
    paths, and the output video/heatmap locations.

    `player_model` picks which player-detector architecture to run (see
    configs.PLAYER_MODEL_REGISTRY / REPORT.md §3 for the real mAP numbers
    behind each option) — falls back to the default if given an unknown id.
    """
    run_dir, run_number = create_run_dir(output_root)
    run_id = f"run_{run_number}"

    player_detector_path = get_player_detector_path(player_model)

    video_frames = read_video(input_video_path)

    # Stubs are cached per input video (by basename) so switching between
    # videos doesn't overwrite another video's cache. Player tracks + team
    # assignment are ALSO cached per player_model (different architecture
    # = different detections = different track_ids) — court keypoints stay
    # video-only since they come from a completely separate detector that
    # doesn't depend on player_model at all, no need to redo them on a
    # model switch.
    video_stem = os.path.splitext(os.path.basename(input_video_path))[0]
    video_stub_dir = os.path.join(stub_root, video_stem)
    model_stub_dir = os.path.join(video_stub_dir, player_model)

    # Detect + Track Players
    player_tracker = PlayerTracker(player_detector_path, tracker_config=PLAYER_TRACKER_CONFIG_PATH)
    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=os.path.join(model_stub_dir, 'player_track_stubs.pkl')
    )

    # Detect Court Keypoints
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(
        video_frames,
        read_from_stub=True,
        stub_path=os.path.join(video_stub_dir, 'court_key_points_stub.pkl')
    )

    # Assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames,
        player_tracks,
        read_from_stub=True,
        stub_path=os.path.join(model_stub_dir, 'player_assignment_stub.pkl')
    )

    # Tactical (top-down) View
    tactical_view_converter = TacticalViewConverter(court_image_path=COURT_IMAGE_PATH)
    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    tactical_player_positions, homography_valid = tactical_view_converter.transform_players_to_tactical_view(
        court_keypoints_per_frame, player_tracks
    )

    # Court-boundary filter: drop tracks whose foot position falls outside
    # the actual court (e.g. bench players, staff on the sideline) so the
    # output video doesn't box people who were never on the court. Only
    # applied on frames where homography was actually computed — on a
    # frame where keypoint detection failed, fall back to the raw tracks
    # instead of hiding every player.
    on_court_player_tracks = [
        (
            {pid: data for pid, data in frame_tracks.items() if pid in tactical_player_positions[idx]}
            if homography_valid[idx]
            else frame_tracks
        )
        for idx, frame_tracks in enumerate(player_tracks)
    ]

    all_player_ids = {pid for frame_tracks in on_court_player_tracks for pid in frame_tracks}

    # Per-player + per-team workrate heatmaps (whole clip), one color per team
    heatmap_dir = os.path.join(run_dir, 'heatmaps')
    heatmap_generator = HeatmapGenerator(
        court_width=tactical_view_converter.width,
        court_height=tactical_view_converter.height,
        court_image_path=tactical_view_converter.court_image_path,
        team_colors=TEAM_COLORS,
    )
    player_heatmaps = heatmap_generator.generate(
        tactical_player_positions, player_assignment, on_court_player_tracks, video_frames, heatmap_dir
    )

    # Draw + write output video one frame at a time (streaming) — see
    # main.py's history / DISCOVERY_REPORT.md for why: building a full
    # second copy of every frame per drawer runs out of memory on
    # long/high-res clips.
    player_tracks_drawer = PlayerTracksDrawer()
    court_keypoint_drawer = CourtKeypointsDrawer()
    frame_number_drawer = FrameNumberDrawer()
    tactical_view_drawer = TacticalViewDrawer()

    interpolated_tactical_positions = tactical_view_drawer.prepare_positions(tactical_player_positions)

    video_name = os.path.splitext(os.path.basename(input_video_path))[0] + '_output.mp4'
    output_video_path = os.path.join(run_dir, video_name)
    fps = get_video_fps(input_video_path)
    height, width = video_frames[0].shape[:2]
    video_writer = StreamingVideoWriter(output_video_path, fps, width, height)

    num_frames = len(video_frames)
    for idx in range(num_frames):
        frame = video_frames[idx]
        frame = player_tracks_drawer.draw_frame(frame, on_court_player_tracks[idx], player_assignment[idx])
        frame = court_keypoint_drawer.draw_frame(frame, court_keypoints_per_frame[idx])
        frame = frame_number_drawer.draw_frame(frame, idx)
        frame = tactical_view_drawer.draw_frame(
            frame, idx,
            tactical_view_converter.court_image_path,
            tactical_view_converter.width,
            tactical_view_converter.height,
            tactical_view_converter.key_points,
            interpolated_tactical_positions,
            player_assignment,
        )
        video_writer.write(frame)
        video_frames[idx] = None  # free this frame's memory now that it's written

    video_writer.release()

    run_info = {
        "run_id": run_id,
        "run_dir": run_dir,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "input_video": input_video_path,
        "num_frames": num_frames,
        "num_players_detected": len(all_player_ids),
        "models": {
            "format": MODEL_FORMAT,
            "player_model": player_model,
            "player_model_label": PLAYER_MODEL_REGISTRY.get(player_model, {}).get('label', player_model),
            "player_detector": PLAYER_MODEL_REGISTRY.get(player_model, {}).get('label', player_model),
            "court_keypoint_detector": COURT_KEYPOINT_DETECTOR_NAME,
        },
        "output_video": output_video_path,
        "heatmap_dir": heatmap_dir,
        "player_heatmaps": player_heatmaps,
    }
    with open(os.path.join(run_dir, 'run_info.json'), 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2)

    return run_info
