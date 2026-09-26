import argparse

from pipeline import run_analysis
from configs import STUBS_DEFAULT_PATH, OUTPUT_ROOT_DIR, DEFAULT_PLAYER_MODEL, PLAYER_MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Player Tracking & Heatmap Analysis')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_root', type=str, default=OUTPUT_ROOT_DIR,
                         help='Root folder under which each run gets its own run_N/ subfolder')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                         help='Path to stub directory')
    parser.add_argument('--player_model', type=str, default=DEFAULT_PLAYER_MODEL,
                         choices=list(PLAYER_MODEL_REGISTRY.keys()),
                         help='Player detector architecture to use (see REPORT.md §3 for mAP per option)')
    return parser.parse_args()


def main():
    args = parse_args()
    run_info = run_analysis(
        args.input_video,
        output_root=args.output_root,
        stub_root=args.stub_path,
        player_model=args.player_model,
    )

    print(f"Model: {run_info['models']['player_model_label']} ({run_info['models']['player_model']})")
    print(f"Run folder: {run_info['run_dir']}")
    print(f"Output video saved to: {run_info['output_video']}")
    print(f"Heatmaps saved under: {run_info['heatmap_dir']}")


if __name__ == '__main__':
    main()
