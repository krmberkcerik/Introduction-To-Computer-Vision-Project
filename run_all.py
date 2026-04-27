import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def run_command(command: List[str]) -> None:
    print(f"[INFO] Running: {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}")


def find_sample_video(test_dir: Path) -> Path:
    videos = [p for p in test_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        raise FileNotFoundError(f"No sample video found under: {test_dir}")
    return sorted(videos)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate and run demo analysis in sequence.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable.")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model path for training.")
    parser.add_argument("--epochs", type=int, default=8, help="Epoch count for fast training.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size for training.")
    parser.add_argument("--runs-dir", type=str, default="runs/train", help="Training run directory.")
    parser.add_argument("--test-dir", type=str, default="test", help="Test directory for evaluation/videos.")
    parser.add_argument("--report-base", type=str, default="reports", help="Base report directory.")
    parser.add_argument("--demo-video", type=str, default=None, help="Optional explicit demo video path.")
    parser.add_argument("--distance-threshold", type=float, default=120.0, help="Release threshold in pixels.")
    parser.add_argument("--max-shot-frames", type=int, default=120, help="Max frames before marking MISS.")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_base) / stamp
    analysis_dir = report_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        args.python,
        "train.py",
        "--data",
        args.data,
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--imgsz",
        str(args.imgsz),
        "--project",
        args.runs_dir,
        "--name",
        f"basketball_free_throw_{stamp}",
    ]
    run_command(train_cmd)

    best_pt = Path("best.pt")
    if not best_pt.exists():
        raise FileNotFoundError("best.pt was not created after training.")

    eval_cmd = [
        args.python,
        "evaluate.py",
        "--model",
        str(best_pt),
        "--runs-dir",
        args.runs_dir,
        "--test-dir",
        args.test_dir,
        "--report-dir",
        str(report_dir),
    ]
    run_command(eval_cmd)

    if args.demo_video:
        demo_video = Path(args.demo_video)
    else:
        demo_video = find_sample_video(Path(args.test_dir))
    if not demo_video.exists():
        raise FileNotFoundError(f"Demo video not found: {demo_video}")

    demo_cmd = [
        args.python,
        "main.py",
        "--model",
        str(best_pt),
        "--input",
        str(demo_video),
        "--output",
        f"demo_{demo_video.stem}.mp4",
        "--output-dir",
        str(analysis_dir),
        "--distance-threshold",
        str(args.distance_threshold),
        "--max-shot-frames",
        str(args.max_shot_frames),
    ]
    run_command(demo_cmd)

    print("[INFO] Pipeline completed successfully.")
    print(f"[INFO] Reports saved under: {report_dir.resolve()}")
    print(f"[INFO] Demo output directory: {analysis_dir.resolve()}")


if __name__ == "__main__":
    main()
