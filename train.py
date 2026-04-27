import argparse
import shutil
from pathlib import Path

import torch  # pyright: ignore[reportMissingImports]
from ultralytics import YOLO  # pyright: ignore[reportMissingImports]


def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for basketball free-throw analysis.")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to YOLO data.yaml file.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model checkpoint.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size.")
    parser.add_argument("--project", type=str, default="runs/train", help="Training runs directory.")
    parser.add_argument("--name", type=str, default="basketball_free_throw", help="Run name.")
    args = parser.parse_args()

    device = select_device()
    print(f"[INFO] Selected device: {device}")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="AdamW",
        patience=2,
        device=device,
        project=args.project,
        name=args.name,
        pretrained=True,
        verbose=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    root_best = Path("best.pt")
    if best_weights.exists():
        shutil.copy2(best_weights, root_best)
        print(f"[INFO] Copied best weights to: {root_best.resolve()}")
    else:
        raise FileNotFoundError(f"best.pt not found at expected location: {best_weights}")


if __name__ == "__main__":
    main()
