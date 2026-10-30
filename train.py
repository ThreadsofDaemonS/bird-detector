"""Fine-tune a pretrained YOLO26n model to detect a single class: bird.

Usage:
    python train.py
    python train.py --epochs 30 --imgsz 640 --batch 16
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="datasets/birds/data.yaml", help="dataset config (see prepare_dataset.py)")
    p.add_argument("--model", default="yolo26n.pt", help="pretrained weights to start from")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu; default: auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="bird")
    p.add_argument("--out", default="weights/best.pt", help="where to copy the best weights")
    args = p.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} not found. Run `python prepare_dataset.py` first.")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=0,
    )

    # Validate the best checkpoint on the validation split and print the metrics.
    best = Path(model.trainer.best)
    metrics = YOLO(best).val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"precision={metrics.box.mp:.3f} recall={metrics.box.mr:.3f} "
          f"mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"Best weights saved to {out}")


if __name__ == "__main__":
    main()
