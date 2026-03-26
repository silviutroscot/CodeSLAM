from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .types import FrameBatch


def _normalize_intensity(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return image[None, ...]


def _resize_pil(path: Path, size: tuple[int, int], mode: str) -> np.ndarray:
    with Image.open(path) as image:
        array = image.convert(mode).resize(size, resample=Image.BILINEAR if mode == "L" else Image.NEAREST)
        return np.asarray(array)


class SceneNetRGBDDataset(Dataset):
    """Dataset loader for SceneNet RGB-D exports used in the paper."""

    def __init__(
        self,
        root: str | Path,
        *,
        image_size: tuple[int, int] = (256, 192),
        depth_scale: float = 1000.0,
        photo_pattern: str = "photo/*.jpg",
        depth_pattern: str = "depth/*.png",
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.samples = self._discover_samples(photo_pattern, depth_pattern)

    def _candidate_depth_paths(self, photo_path: Path) -> list[Path]:
        stem = photo_path.stem
        relative_root = photo_path.parent.parent
        candidates = [relative_root / "depth" / f"{stem}.png"]
        for suffix in ("_intensity", "_resized"):
            if stem.endswith(suffix):
                candidates.append(relative_root / "depth" / f"{stem[: -len(suffix)]}.png")
        return candidates

    def _discover_samples(self, photo_pattern: str, depth_pattern: str) -> list[tuple[Path, Path]]:
        samples = []
        discovered_depths = {path.resolve() for path in self.root.rglob(depth_pattern)}
        for photo_path in sorted(self.root.rglob(photo_pattern)):
            for depth_path in self._candidate_depth_paths(photo_path):
                if depth_path.resolve() in discovered_depths:
                    samples.append((photo_path, depth_path))
                    break
        if not samples:
            raise FileNotFoundError(f"No paired photo/depth files found under {self.root}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        photo_path, depth_path = self.samples[index]
        width, height = self.image_size
        intensity = _resize_pil(photo_path, (width, height), "L")
        depth = _resize_pil(depth_path, (width, height), "I")
        depth = depth.astype(np.float32) / self.depth_scale

        return {
            "frame_id": photo_path.stem,
            "intensity": torch.from_numpy(_normalize_intensity(intensity)),
            "depth": torch.from_numpy(depth[None, ...]),
        }


class FolderSequenceDataset(Dataset):
    """Simple sequential image loader for SLAM and pair-optimization demos."""

    def __init__(
        self,
        image_dir: str | Path,
        *,
        image_size: tuple[int, int] = (256, 192),
        pattern: str = "*.png",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.paths = sorted(self.image_dir.glob(pattern))
        if not self.paths:
            self.paths = sorted(self.image_dir.glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> FrameBatch:
        path = self.paths[index]
        width, height = self.image_size
        intensity = _resize_pil(path, (width, height), "L")
        return FrameBatch(
            intensity=torch.from_numpy(_normalize_intensity(intensity)),
            frame_id=path.stem,
        )
