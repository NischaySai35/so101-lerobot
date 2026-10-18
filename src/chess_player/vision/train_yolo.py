"""Train YOLO on data/dataset (capture_dataset.py presets + confirmed board readings). Split by position.

  python -m chess_player.vision.train_yolo                  # fine-tune the pretrained chess model (default, fast)
  python -m chess_player.vision.train_yolo --model yolov8s.pt   # or train from a generic YOLO

Fine-tuning keeps the pretrained model's class layout (labels are remapped to it), so what it already knows about
chess pieces is kept and it only has to learn this board, these pieces and this camera angle.
Copies the best weights to data/yolo/best.pt, which detect_board.py then prefers over the pretrained model.
"""
import argparse
import json
import random
import shutil

from chess_player import config as C
from chess_player.vision.detect_board import CLASSES

DS = C.DATA / "dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--model", default=str(C.YOLO_PRETRAINED))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=960, help="webcam frames are 1280x720 with small pieces: >640 helps")
    a = ap.parse_args()

    by_pos = {}
    for meta in sorted((DS / "meta").glob("*.json")):
        by_pos.setdefault(json.loads(meta.read_text())["position"], []).append(DS / "images" / f"{meta.stem}.jpg")
    positions = sorted(by_pos)
    if len(positions) < 3:
        raise SystemExit(f"only {len(positions)} positions captured; need several (aim for 20-30)")
    random.Random(0).shuffle(positions)
    n_val = max(1, round(len(positions) * a.val_frac))
    val, train = positions[:n_val], positions[n_val:]

    # class layout: the starting model's (fine-tuning keeps its head); labels on disk use CLASSES order
    from ultralytics import YOLO

    from chess_player.vision.detect_board import _symbol

    base = YOLO(a.model)
    names = dict(base.names) if not str(a.model).startswith("yolov8") else {i: c for i, c in enumerate(CLASSES)}
    sym_to_id = {_symbol(n): i for i, n in names.items() if _symbol(n)}
    remap = {i: sym_to_id[c] for i, c in enumerate(CLASSES)}
    work = C.DATA / "yolo_work"
    if work.exists():
        shutil.rmtree(work)
    for split, keys in (("train", train), ("val", val)):
        (work / split / "images").mkdir(parents=True)
        (work / split / "labels").mkdir(parents=True)
        for k in keys:
            for img in by_pos[k]:
                lab = DS / "labels" / f"{img.stem}.txt"
                (work / split / "images" / img.name).symlink_to(img)
                lines = []
                for ln in (lab.read_text().split("\n") if lab.exists() else []):
                    if ln.strip():
                        cid, *rest = ln.split()
                        lines.append(" ".join([str(remap[int(cid)]), *rest]))
                (work / split / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
    yaml = work / "chess.yaml"
    yaml.write_text(f"path: {work}\ntrain: train/images\nval: val/images\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in names.items()))
    print(f"train positions {train}\nval positions {val}\nstarting from {a.model}")

    res = base.train(
        data=str(yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, project=str(C.DATA / "yolo_runs"), name="chess",
        fliplr=0.0,  # mirrored boards/pieces don't occur; keep orientation cues
        mosaic=0.5, degrees=3.0, hsv_v=0.4, patience=20, exist_ok=True,
    )
    best = C.Path(res.save_dir) / "weights" / "best.pt"
    C.YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, C.YOLO_WEIGHTS)
    print(f"copied {best} -> {C.YOLO_WEIGHTS}. Check the confusion matrix in {res.save_dir}")


if __name__ == "__main__":
    main()
