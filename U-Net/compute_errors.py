from __future__ import annotations

import argparse
import glob
import os


SUBFOLDERS = range(0, 1000)
TEST_DIR = "../data/val"


def collect_test_files(test_dir: str = TEST_DIR) -> tuple[list[str], list[str]]:
    test_paths = [os.path.join(test_dir, str(folder)) for folder in SUBFOLDERS]

    test_img_paths: list[str] = []
    test_label_paths: list[str] = []
    for path in test_paths:
        test_img_paths.extend(sorted(glob.iglob(os.path.join(path, "photo", "*"), recursive=True)))
        test_label_paths.extend(sorted(glob.iglob(os.path.join(path, "depth", "*"), recursive=True)))
    return test_img_paths, test_label_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the legacy U-Net validation split.")
    parser.add_argument("--test-dir", default=TEST_DIR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    images, labels = collect_test_files(args.test_dir)
    print(f"images={len(images)} labels={len(labels)}")
    if images:
        print(f"first_image={images[0]}")
    if labels:
        print(f"first_depth={labels[0]}")


if __name__ == "__main__":
    main()
