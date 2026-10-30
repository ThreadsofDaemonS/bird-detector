"""Build a small single-class (bird) YOLO dataset from COCO 2017.

Downloads the COCO annotations, selects images containing birds, downloads only
those images, converts boxes to YOLO format and splits into train/val.

Usage:
    python prepare_dataset.py --num-images 1200
"""
import argparse
import json
import random
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}/{name}"
BIRD_CATEGORY = "bird"


def download(url: str, dest: Path, progress: bool = True) -> None:
    """Stream a file to disk with a progress bar (skipped if it already exists)."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, disable=not progress) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def load_bird_images(zip_path: Path, split: str, min_box_frac: float):
    """Return {file_name: [(cx, cy, w, h), ...]} normalized YOLO boxes of birds."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"annotations/instances_{split}.json") as f:
            data = json.load(f)
    bird_id = next(c["id"] for c in data["categories"] if c["name"] == BIRD_CATEGORY)
    images = {im["id"]: im for im in data["images"]}
    result: dict[str, list] = {}
    for ann in data["annotations"]:
        if ann["category_id"] != bird_id or ann.get("iscrowd", 0):
            continue
        im = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if w * h < min_box_frac * im["width"] * im["height"]:
            continue  # skip tiny / unreliable boxes
        box = ((x + w / 2) / im["width"], (y + h / 2) / im["height"], w / im["width"], h / im["height"])
        result.setdefault(im["file_name"], []).append(box)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("datasets/birds"))
    p.add_argument("--num-images", type=int, default=1200)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--min-box-frac", type=float, default=0.0005,
                   help="drop boxes smaller than this fraction of the image area")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cache = args.out.parent / "_cache"
    zip_path = cache / "annotations_trainval2017.zip"
    download(ANN_URL, zip_path)

    # Use train2017 annotations for the pool; we make our own train/val split.
    pool = load_bird_images(zip_path, "train2017", args.min_box_frac)
    names = sorted(pool)
    random.Random(args.seed).shuffle(names)
    names = names[: args.num_images]
    n_val = int(len(names) * args.val_split)
    splits = {"val": names[:n_val], "train": names[n_val:]}
    print(f"Pool: {len(pool)} bird images, using {len(names)} "
          f"(train={len(splits['train'])}, val={len(splits['val'])})")

    if args.out.exists():
        shutil.rmtree(args.out)

    def fetch(item):
        split, name = item
        img_dest = args.out / "images" / split / name
        download(IMG_URL.format(split="train2017", name=name), img_dest, progress=False)
        lbl = args.out / "labels" / split / (Path(name).stem + ".txt")
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in pool[name]))

    jobs = [(s, n) for s, ns in splits.items() for n in ns]
    with ThreadPoolExecutor(8) as ex:
        list(tqdm(ex.map(fetch, jobs), total=len(jobs), desc="images"))

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: bird\n"
    )
    print(f"Done. Dataset config: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
