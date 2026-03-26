from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codeslam.config import ModelConfig
from codeslam.dataset import SceneNetRGBDDataset
from codeslam.network import CodeSLAMDepthModel
from codeslam.training import evaluate_zero_code_prior


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-image CodeSLAM depth prediction on an RGB-D dataset.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = ModelConfig(**checkpoint["model_config"]) if "model_config" in checkpoint else ModelConfig()
    model = CodeSLAMDepthModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = SceneNetRGBDDataset(
        args.data_root,
        image_size=(model_config.input_width, model_config.input_height),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics = evaluate_zero_code_prior(model.to(device), loader, device)
    print(metrics)


if __name__ == "__main__":
    main()
