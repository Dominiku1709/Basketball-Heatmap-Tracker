"""
run_utils.py

Manages numbered run folders under output/ (output/run_1/, output/run_2/, ...)
so each `python main.py` invocation gets its own folder for the output
video, heatmaps, and run_info.json instead of overwriting the previous run.
"""

import os
import re

RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")


def create_run_dir(output_root: str = "output") -> tuple[str, int]:
    """
    Creates the next numbered run_N folder under `output_root` and returns
    its path + run number. `output_root` is created if missing.
    """
    os.makedirs(output_root, exist_ok=True)

    existing_run_numbers = [
        int(match.group(1))
        for entry in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, entry))
        for match in [RUN_DIR_PATTERN.match(entry)]
        if match
    ]

    next_run_number = max(existing_run_numbers, default=0) + 1
    run_dir = os.path.join(output_root, f"run_{next_run_number}")
    os.makedirs(run_dir, exist_ok=True)

    return run_dir, next_run_number
