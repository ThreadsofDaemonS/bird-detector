"""Run the bird detector on every frame of a video and save the annotated result.

Usage:
    python predict_video.py --input examples/input.mp4 --output examples/output.mp4
"""
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Prefer H.264 (plays in browsers / GitHub); fall back to mp4v if unavailable."""
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Cannot open video writer for {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="path to the input video")
    p.add_argument("--output", required=True, help="path of the annotated output video (.mp4)")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu; default: auto")
    args = p.parse_args()

    if not Path(args.input).is_file():
        raise SystemExit(f"Input video not found: {args.input}")
    if not Path(args.weights).is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(out_path, fps, size)
    model = YOLO(args.weights)

    detections = 0
    try:
        with tqdm(total=total or None, desc="frames", unit="f") as bar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                       device=args.device, verbose=False)[0]
                detections += len(result.boxes)
                # plot() draws boxes with "<class name> <confidence>" labels.
                writer.write(result.plot())
                bar.update(1)
    finally:
        cap.release()
        writer.release()

    print(f"Saved {out_path} ({bar.n} frames, {detections} detections)")


if __name__ == "__main__":
    main()
