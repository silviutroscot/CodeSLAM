from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codeslam.config import CameraModel, ModelConfig
from codeslam.dataset import FolderSequenceDataset
from codeslam.network import CodeSLAMDepthModel
from codeslam.system import CodeSLAMSystem


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sliding-window monocular CodeSLAM pipeline on an image folder.")
    parser.add_argument("image_dir")
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

    dataset = FolderSequenceDataset(args.image_dir, image_size=(model.config.input_width, model.config.input_height))
    camera = CameraModel(args.fx, args.fy, args.cx, args.cy, model.config.input_width, model.config.input_height)
    system = CodeSLAMSystem(model, camera)

    for frame in dataset:
        result = system.process_frame(frame)
        print(frame.frame_id, result.pose_w_c.reshape(-1).tolist(), "keyframe" if result.inserted_keyframe else "frame")


if __name__ == "__main__":
    main()
