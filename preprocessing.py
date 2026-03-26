from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import statistics
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


TRAINING_SET_PATH = "data/train/"
DATASET_SUBFOLDERS = range(0, 15)
IMAGE_NEW_WIDTH = 256
IMAGE_NEW_HEIGHT = 192


def create_intensity_images_from_rgb_images_folder(path: str, subfolder: int) -> None:
    if Image is None:
        raise ImportError("Pillow is required for image preprocessing.")
    for filename in glob.iglob(f"{path}{subfolder}/*/photo/*", recursive=True):
        if filename.endswith("_intensity.jpg"):
            continue
        img = Image.open(filename).convert("L")
        image_name_without_extension = filename.split(".")[0]
        intensity_image_name = image_name_without_extension + "_intensity.jpg"
        img.save(intensity_image_name)


def resize_intensity_images(path: str, new_width: int, new_height: int, subfolder: int) -> None:
    if Image is None:
        raise ImportError("Pillow is required for image preprocessing.")
    for filename in glob.iglob(f"{path}{subfolder}/*/photo/*_intensity.jpg", recursive=True):
        img = Image.open(filename)
        resized_image = img.resize((new_width, new_height))
        image_name_without_extension = filename.split(".")[0]
        resized_intensity_image_name = image_name_without_extension + "_resized.jpg"
        resized_image.save(resized_intensity_image_name, "JPEG", optimize=True)
        os.remove(filename)


def scale_depth(image: list[float], average: float) -> None:
    for index in [0, len(image) - 1]:
        image[index] = average / (average + image[index])


def normalize_depth_values(path: str, subfolder: int) -> None:
    if Image is None or np is None:
        raise ImportError("Pillow and numpy are required for depth normalization.")
    for filename in glob.iglob(f"{path}{subfolder}/*/depth/*[0-9].png", recursive=True):
        img = Image.open(filename)
        size = img.size
        image_values = img.histogram()
        average_depth = statistics.mean(image_values)
        if average_depth == 0:
            average_depth = 1e-6
        scale_depth(image_values, average_depth)
        image_array = np.array(image_values, dtype=np.float32)
        image = Image.new("L", size)
        image.putdata(image_array)
        normalized_image_name = filename.split(".")[0] + "_normalized.png"
        image.save(normalized_image_name, "PNG", optimize=True)


def remove_normalized_depth_images(path: str) -> None:
    for filename in glob.iglob(path + "**/*/depth/*_normalized.png", recursive=True):
        os.remove(filename)


def remove_intensity_images(path: str) -> None:
    for filename in glob.iglob(path + "**/*/photo/*_intensity.jpg", recursive=True):
        os.remove(filename)
    for filename in glob.iglob(path + "**/*/photo/*_resized.jpg", recursive=True):
        os.remove(filename)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SceneNet RGB-D exports for CodeSLAM training.")
    parser.add_argument(
        "actions",
        nargs="+",
        choices=["i", "r", "n", "clean-intensity", "clean-depth"],
        help="i=create intensity, r=resize intensity, n=normalize depth",
    )
    parser.add_argument("--data-root", default=TRAINING_SET_PATH)
    parser.add_argument("--width", type=int, default=IMAGE_NEW_WIDTH)
    parser.add_argument("--height", type=int, default=IMAGE_NEW_HEIGHT)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    path = args.data_root
    Path(path).mkdir(parents=True, exist_ok=True)

    with mp.Pool(args.workers) as pool:
        if "i" in args.actions:
            pool.starmap(
                create_intensity_images_from_rgb_images_folder,
                [(path, subfolder) for subfolder in DATASET_SUBFOLDERS],
            )
        if "r" in args.actions:
            pool.starmap(
                resize_intensity_images,
                [(path, args.width, args.height, subfolder) for subfolder in DATASET_SUBFOLDERS],
            )
        if "n" in args.actions:
            pool.starmap(
                normalize_depth_values,
                [(path, subfolder) for subfolder in DATASET_SUBFOLDERS],
            )

    if "clean-intensity" in args.actions:
        remove_intensity_images(path)
    if "clean-depth" in args.actions:
        remove_normalized_depth_images(path)


if __name__ == "__main__":
    main()
