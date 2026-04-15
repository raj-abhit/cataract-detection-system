from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def class_from_label_file(label_path: Path) -> int:
    classes: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        classes.append(int(line.split()[0]))
    if not classes:
        raise ValueError(f"Empty label file: {label_path}")
    return Counter(classes).most_common(1)[0][0]


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_groups(items: list[str], train_ratio: float, val_ratio: float) -> tuple[set[str], set[str], set[str]]:
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n >= 3 and n_test == 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
    train = set(items[:n_train])
    val = set(items[n_train : n_train + n_val])
    test = set(items[n_train + n_val : n_train + n_val + n_test])
    return train, val, test


def main() -> None:
    random.seed(SEED)

    root = Path("cataract.yolov12")
    src_images = root / "train" / "images"
    src_labels = root / "train" / "labels"

    if not src_images.exists() or not src_labels.exists():
        raise FileNotFoundError("Expected source folders at cataract.yolov12/train/images and .../labels")

    out_root = Path("cataract.yolov12_ready")
    for split in ("train", "valid", "test"):
        ensure_clean_dir(out_root / split / "images")
        ensure_clean_dir(out_root / split / "labels")

    group_to_files: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    group_class_votes: dict[str, list[int]] = defaultdict(list)

    for image_path in src_images.glob("*"):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = src_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        group_id = image_path.name.split(".rf.")[0]
        group_to_files[group_id].append((image_path, label_path))
        group_class_votes[group_id].append(class_from_label_file(label_path))

    group_to_class: dict[str, int] = {}
    for gid, votes in group_class_votes.items():
        group_to_class[gid] = Counter(votes).most_common(1)[0][0]

    groups_by_class: dict[int, list[str]] = defaultdict(list)
    for gid, cls in group_to_class.items():
        groups_by_class[cls].append(gid)

    for cls in groups_by_class:
        random.shuffle(groups_by_class[cls])

    split_groups_map = {"train": set(), "valid": set(), "test": set()}
    for cls, gids in groups_by_class.items():
        tr, va, te = split_groups(gids, TRAIN_RATIO, VAL_RATIO)
        split_groups_map["train"].update(tr)
        split_groups_map["valid"].update(va)
        split_groups_map["test"].update(te)

    for split, gids in split_groups_map.items():
        for gid in gids:
            for image_path, label_path in group_to_files[gid]:
                shutil.copy2(image_path, out_root / split / "images" / image_path.name)
                shutil.copy2(label_path, out_root / split / "labels" / label_path.name)

    data_yaml = """path: .
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['Cataract', 'Normal']
"""
    (out_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    src_readme = root / "README.roboflow.txt"
    if src_readme.exists():
        shutil.copy2(src_readme, out_root / src_readme.name)

    for split in ("train", "valid", "test"):
        image_count = sum(1 for _ in (out_root / split / "images").glob("*"))
        label_count = sum(1 for _ in (out_root / split / "labels").glob("*.txt"))
        print(f"{split}: images={image_count}, labels={label_count}")


if __name__ == "__main__":
    main()
