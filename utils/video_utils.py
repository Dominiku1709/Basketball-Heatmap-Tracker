"""
video_utils.py

Utility functions for reading and saving videos using OpenCV.
"""

import cv2
import os
import subprocess
from typing import List

import imageio_ffmpeg


def read_video(video_path: str) -> List:
    """
    Reads a video file and returns its frames as a list of numpy arrays.

    Holds the whole video in memory — fine for short clips, but the caller
    is responsible for not also holding multiple full-length copies at once
    for long/high-res videos (see StreamingVideoWriter / main.py, which
    frees each frame as soon as it's drawn+written instead of building
    successive whole-video copies per drawer).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def get_video_fps(video_path: str, default_fps: float = 24.0) -> float:
    """
    Reads a video's FPS from its container metadata. Falls back to
    `default_fps` if the file can't report one (e.g. 0/invalid, some
    container quirks) — matches the previous hardcoded behaviour instead of
    silently producing a 0-fps (unplayable) file.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default_fps


def save_video(output_frames, output_path, fps: float = 24.0):
    """
    Writes an already-fully-materialized list of frames to an mp4 file.

    Kept for callers that build the whole output list up front. For long
    videos, prefer StreamingVideoWriter so frames are written (and freed)
    one at a time instead of needing every drawer to hold a full-length
    copy of the video simultaneously.
    """
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    if not output_path.endswith('.mp4'):
        output_path += '.mp4'

    height, width = output_frames[0].shape[:2]
    writer = StreamingVideoWriter(output_path, fps, width, height)
    for frame in output_frames:
        writer.write(frame)
    writer.release()


class StreamingVideoWriter:
    """
    Writes frames one at a time as they're produced (so a caller never needs
    to hold the full output video in memory as a Python list — see main.py's
    render loop), encoding to H.264 via a piped ffmpeg process instead of
    cv2.VideoWriter.

    Why: opencv-python(-headless) wheels ship without a working H.264
    encoder (OpenH264 isn't bundled — a licensing thing), so
    cv2.VideoWriter falls back silently to MPEG-4 Part 2 ("mp4v"/FMP4).
    That file is a valid .mp4 and plays fine in players with a general
    decoder (VLC, OpenCV itself, ffplay) — but browsers only decode H.264/
    VP9/AV1 for HTML5 <video>, so it silently fails to play in the web UI.
    `imageio-ffmpeg` bundles a static ffmpeg build with libx264, so we pipe
    raw BGR frames into `ffmpeg -c:v libx264` ourselves.
    """

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        dir_name = os.path.dirname(output_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        if not output_path.endswith('.mp4'):
            output_path += '.mp4'

        self.output_path = output_path
        self.width = width
        self.height = height

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-',
            '-an',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast',
            '-movflags', '+faststart',  # moov atom up front — lets the browser start playing before the whole file downloads
            output_path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, frame) -> None:
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))
        self._proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        self._proc.stdin.close()
        stderr = self._proc.stderr.read()
        self._proc.stderr.close()
        return_code = self._proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed encoding {self.output_path} (exit {return_code}): {stderr.decode(errors='replace')}")
        print(f"Video saved to {self.output_path}")
