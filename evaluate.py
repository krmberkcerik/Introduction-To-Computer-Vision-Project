import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd  # pyright: ignore[reportMissingImports]

from free_throw_analyzer import FreeThrowAnalyzer


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_latest_train_run(train_runs_dir: Path) -> Optional[Path]:
    if not train_runs_dir.exists():
        return None
    candidates = [p for p in train_runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def export_training_report(run_dir: Path, report_dir: Path) -> Dict[str, Optional[str]]:
    report_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Optional[str]] = {"results_csv": None, "confusion_matrix": None}

    results_src = run_dir / "results.csv"
    cm_src_candidates = [
        run_dir / "confusion_matrix_normalized.png",
        run_dir / "confusion_matrix.png",
    ]
    cm_src = next((p for p in cm_src_candidates if p.exists()), None)

    results_dst = report_dir / "results.csv"
    if copy_if_exists(results_src, results_dst):
        out["results_csv"] = str(results_dst.resolve())

    if cm_src is not None:
        cm_dst = report_dir / cm_src.name
        if copy_if_exists(cm_src, cm_dst):
            out["confusion_matrix"] = str(cm_dst.resolve())

    return out


def parse_ground_truth(sidecar_file: Path) -> Optional[bool]:
    if not sidecar_file.exists():
        return None
    text = sidecar_file.read_text(encoding="utf-8").strip().lower()
    if text in {"1", "make", "basket", "hit", "true"}:
        return True
    if text in {"0", "miss", "false"}:
        return False
    return None


def evaluate_test_videos(
    model_path: str,
    test_dir: Path,
    distance_threshold: float,
    max_shot_frames: int,
) -> Tuple[List[Dict[str, object]], Optional[float]]:
    videos = [p for p in test_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
    rows: List[Dict[str, object]] = []

    if not videos:
        return rows, None

    gt_total = 0
    gt_correct = 0

    for video in sorted(videos):
        analyzer = FreeThrowAnalyzer(
            model_path=model_path,
            release_distance_threshold=distance_threshold,
            max_shot_frames=max_shot_frames,
        )
        metrics = analyzer.analyze_video(str(video), output_video=None, display=False)
        predicted_make = metrics["basket"] > metrics["miss"]

        gt = parse_ground_truth(video.with_suffix(".txt"))
        is_correct = None
        if gt is not None:
            gt_total += 1
            is_correct = predicted_make == gt
            if is_correct:
                gt_correct += 1

        rows.append(
            {
                "video": str(video),
                "predicted_make": predicted_make,
                "basket": metrics["basket"],
                "miss": metrics["miss"],
                "total_shots": metrics["total_shots"],
                "accuracy_percent": metrics["accuracy_percent"],
                "ground_truth_make": gt,
                "match": is_correct,
            }
        )

    overall_accuracy = (gt_correct / gt_total) * 100.0 if gt_total else None
    return rows, overall_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate training outputs and test videos.")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to trained model weights.")
    parser.add_argument("--runs-dir", type=str, default="runs/train", help="Directory where training runs are stored.")
    parser.add_argument("--test-dir", type=str, default="test", help="Directory containing test videos.")
    parser.add_argument("--report-dir", type=str, default="reports", help="Base directory for exported reports.")
    parser.add_argument("--timestamped", action="store_true", help="Create date-stamped report subfolder.")
    parser.add_argument("--distance-threshold", type=float, default=120.0, help="Human-ball release threshold in pixels.")
    parser.add_argument("--max-shot-frames", type=int, default=120, help="Max frames before shot is MISS.")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if args.timestamped:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = report_dir / stamp
    train_run = find_latest_train_run(Path(args.runs_dir))
    training_artifacts = {"results_csv": None, "confusion_matrix": None}

    if train_run is not None:
        training_artifacts = export_training_report(train_run, report_dir / "training")
        if training_artifacts["results_csv"]:
            df = pd.read_csv(training_artifacts["results_csv"])
            summary = {
                "epochs_logged": int(len(df)),
                "final_metrics": df.iloc[-1].to_dict() if not df.empty else {},
            }
            (report_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows, overall_accuracy = evaluate_test_videos(
        model_path=args.model,
        test_dir=Path(args.test_dir),
        distance_threshold=args.distance_threshold,
        max_shot_frames=args.max_shot_frames,
    )
    test_report_path = report_dir / "test_video_evaluation.csv"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(test_report_path, index=False)

    final_report = {
        "training_artifacts": training_artifacts,
        "test_video_report_csv": str(test_report_path.resolve()),
        "overall_video_accuracy_percent": None if overall_accuracy is None else round(overall_accuracy, 2),
        "notes": "Overall accuracy is computed only for videos with a sidecar .txt ground-truth file.",
    }
    json_path = report_dir / "evaluation_report.json"
    json_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")

    print("[INFO] Evaluation finished.")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
