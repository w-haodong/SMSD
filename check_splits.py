# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import combinations


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
EXPECTED_COUNTS = {
    "aasce-128": {"train": 431, "val": 50, "test": 128},
    "aasce-98": {"train": 549, "val": 60, "test": 98},
    "clinical-150": {"train": 637, "val": 70, "test": 150},
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_split(data_dir: str, split: str):
    split_root = os.path.join(data_dir, split)
    image_dir = os.path.join(split_root, "images")
    label_dir = os.path.join(split_root, "labels")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    images = [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if name.lower().endswith(IMAGE_EXTENSIONS)
    ]
    records = []
    missing_labels = []
    for image_path in images:
        image_name = os.path.basename(image_path)
        label_candidates = (
            os.path.join(label_dir, image_name + ".mat"),
            os.path.join(label_dir, os.path.splitext(image_name)[0] + ".mat"),
        )
        if not any(os.path.isfile(path) for path in label_candidates):
            missing_labels.append(image_name)
        records.append(
            {
                "name": image_name,
                "sample_id": os.path.splitext(image_name)[0].casefold(),
                "sha256": _sha256(image_path),
            }
        )
    return records, missing_labels


def audit_splits(data_dir: str, protocol: str):
    expected = EXPECTED_COUNTS[protocol]
    split_records = {}
    missing_labels = {}
    for split in ("train", "val", "test"):
        split_records[split], missing_labels[split] = _collect_split(data_dir, split)

    overlaps = []
    for left, right in combinations(("train", "val", "test"), 2):
        left_ids = {item["sample_id"] for item in split_records[left]}
        right_ids = {item["sample_id"] for item in split_records[right]}
        left_hashes = {item["sha256"] for item in split_records[left]}
        right_hashes = {item["sha256"] for item in split_records[right]}
        overlaps.append(
            {
                "splits": [left, right],
                "shared_sample_ids": len(left_ids & right_ids),
                "identical_images": len(left_hashes & right_hashes),
            }
        )

    observed = {split: len(split_records[split]) for split in split_records}
    count_match = observed == expected
    labels_complete = all(not values for values in missing_labels.values())
    disjoint = all(
        item["shared_sample_ids"] == 0 and item["identical_images"] == 0
        for item in overlaps
    )
    return {
        "protocol": protocol,
        "data_dir": os.path.abspath(os.path.normpath(data_dir)),
        "expected_counts": expected,
        "observed_counts": observed,
        "count_match": count_match,
        "labels_complete": labels_complete,
        "missing_labels": missing_labels,
        "overlaps": overlaps,
        "disjoint": disjoint,
        "valid": bool(count_match and labels_complete and disjoint),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify the paper-defined train/validation/test protocol before an experiment."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--protocol", choices=sorted(EXPECTED_COUNTS), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_splits(args.data_dir, args.protocol)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
