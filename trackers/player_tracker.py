"""
player_tracker.py

Detects and tracks players using a YOLO detector combined with Ultralytics'
built-in BoT-SORT tracker (camera motion compensation + Re-ID, configured in
trackers/botsort.yaml). Replaces the previous plain supervision.ByteTrack
setup, which had no camera-motion handling and fragmented IDs under
broadcast-style camera pan.
"""

import os
import sys
from typing import Dict, List, Optional

from ultralytics import YOLO

from utils.stubs_utils import read_stub, save_stub

sys.path.append("..")

DEFAULT_TRACKER_CONFIG = os.path.join(os.path.dirname(__file__), "botsort.yaml")


class PlayerTracker:
    def __init__(self, model_path: str, tracker_config: str = DEFAULT_TRACKER_CONFIG):
        """
        Initializes the PlayerTracker with a YOLO player-detector weight and
        a BoT-SORT tracker config.
        """
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config

    def get_object_tracks(
        self,
        frames: List,
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
        conf: float = 0.5,
        batch_size: int = 20,
    ) -> List[Dict[int, Dict[str, List[float]]]]:
        """
        Tracks players across video frames using YOLO detection + BoT-SORT.

        `persist=True` keeps the tracker's internal state alive across
        successive `.track()` calls, so batching frames for memory doesn't
        break ID continuity as long as batches are processed in order.
        NOTE: not yet validated against a real video/weight — re-check track
        continuity once the player detector weight is available (see
        configs.PLAYER_DETECTOR_PATH).
        """
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        tracks = []
        player_class_id = None

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            results = self.model.track(
                batch,
                conf=conf,
                persist=True,
                tracker=self.tracker_config,
                verbose=False,
            )

            for result in results:
                if player_class_id is None:
                    class_id_map = {v: k for k, v in result.names.items()}
                    player_class_id = class_id_map.get("player")

                frame_tracks: Dict[int, Dict[str, List[float]]] = {}
                boxes = result.boxes

                if boxes is not None and boxes.id is not None:
                    xyxy = boxes.xyxy.tolist()
                    cls_ids = boxes.cls.tolist()
                    track_ids = boxes.id.tolist()

                    for bbox, cls_id, track_id in zip(xyxy, cls_ids, track_ids):
                        if int(cls_id) == player_class_id:
                            frame_tracks[int(track_id)] = {"bbox": bbox}

                tracks.append(frame_tracks)

        if stub_path:
            save_stub(stub_path, tracks)

        return tracks
