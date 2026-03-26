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

from codeslam.config import CameraModel, ModelConfig
from codeslam.network import CodeSLAMDepthModel
from codeslam.system import CodeSLAMSystem
from codeslam.types import FrameBatch


def load_intensity(path: str, width: int, height: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((width, height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jointly optimize the relative pose and latent codes for a frame pair.")
    parser.add_argument("first_image")
    parser.add_argument("second_image")
    parser.add_argument("--checkpoint", help="Optional trained model checkpoint.")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model_config = ModelConfig(**checkpoint["model_config"]) if "model_config" in checkpoint else ModelConfig()
        model = CodeSLAMDepthModel(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = CodeSLAMDepthModel(ModelConfig(input_width=args.width, input_height=args.height))
    model.eval()

    width = model.config.input_width
    height = model.config.input_height
    camera = CameraModel(args.fx, args.fy, args.cx, args.cy, width, height)
    system = CodeSLAMSystem(model, camera)

    first = FrameBatch(intensity=load_intensity(args.first_image, width, height), frame_id="first")
    second = FrameBatch(intensity=load_intensity(args.second_image, width, height), frame_id="second")
    first_keyframe, second_keyframe = system.bootstrap(first, second)
    relative = torch.linalg.inv(first_keyframe.pose_w_c) @ second_keyframe.pose_w_c
    print("first_pose_w_c")
    print(first_keyframe.pose_w_c)
    print("second_pose_w_c")
    print(second_keyframe.pose_w_c)
    print("relative_pose")
    print(relative)


if __name__ == "__main__":
    main()
