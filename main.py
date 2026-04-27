import argparse
from pathlib import Path

from free_throw_analyzer import FreeThrowAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basketball free-throw analysis on a video.")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to trained model weights.")
    parser.add_argument("--input", type=str, required=True, help="Input video path.")
    parser.add_argument("--output", type=str, default="analysis_output.mp4", help="Output annotated video path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory for analysis video.")
    parser.add_argument("--distance-threshold", type=float, default=120.0, help="Human-ball release threshold in pixels.")
    parser.add_argument("--max-shot-frames", type=int, default=120, help="Max frames to wait before shot ends as MISS.")
    parser.add_argument("--display", action="store_true", help="Display real-time window during processing.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path = Path(args.output)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_path.name

    analyzer = FreeThrowAnalyzer(
        model_path=args.model,
        release_distance_threshold=args.distance_threshold,
        max_shot_frames=args.max_shot_frames,
    )
    metrics = analyzer.analyze_video(str(input_path), output_video=str(output_path), display=args.display)
    print("[INFO] Analysis completed.")
    print(f"[INFO] Metrics: {metrics}")
    print(f"[INFO] Output video: {output_path.resolve()}")


if __name__ == "__main__":
    main()
