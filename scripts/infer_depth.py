from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codeslam.config import ModelConfig
from codeslam.network import CodeSLAMDepthModel


def load_intensity(path: str, width: int, height: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((width, height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def save_depth_png(depth: np.ndarray, output_path: Path, max_depth: float) -> None:
    clipped = np.clip(depth, 0.0, max_depth)
    if max_depth <= 0:
        max_depth = 1.0
    image = (255.0 * clipped / max_depth).astype(np.uint8)
    Image.fromarray(image).save(output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a dense depth map from a single image using the CodeSLAM prior.")
    parser.add_argument("image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-prefix", default="prediction")
    parser.add_argument("--max-visualization-depth", type=float, default=10.0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = ModelConfig(**checkpoint["model_config"]) if "model_config" in checkpoint else ModelConfig()
    model = CodeSLAMDepthModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    width = model.config.input_width
    height = model.config.input_height
    intensity = load_intensity(args.image, width, height)
    with torch.no_grad():
        prediction = model(intensity, code=model.zero_code(1, intensity.device, intensity.dtype), sample_posterior=False)

    depth = prediction.depth_pyramid[-1].squeeze().cpu().numpy()
    uncertainty = prediction.scale_pyramid[-1].squeeze().cpu().numpy()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_prefix) + "_depth.npy", depth)
    np.save(str(output_prefix) + "_uncertainty.npy", uncertainty)
    save_depth_png(depth, Path(str(output_prefix) + "_depth.png"), args.max_visualization_depth)
    save_depth_png(uncertainty, Path(str(output_prefix) + "_uncertainty.png"), float(uncertainty.max()) if uncertainty.size else 1.0)


if __name__ == "__main__":
    main()
