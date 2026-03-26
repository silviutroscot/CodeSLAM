from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader, random_split

from codeslam.config import LossConfig, ModelConfig, TrainingConfig
from codeslam.dataset import SceneNetRGBDDataset
from codeslam.network import CodeSLAMDepthModel
from codeslam.training import fit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the conditioned CodeSLAM depth auto-encoder.")
    parser.add_argument("--data-root", required=True, help="Root directory containing paired SceneNet photo/depth folders.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--final-learning-rate", type=float, default=1e-6)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    model_config = ModelConfig()
    dataset = SceneNetRGBDDataset(
        args.data_root,
        image_size=(model_config.input_width, model_config.input_height),
    )
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = max(1, len(dataset) - val_size)
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    model = CodeSLAMDepthModel(model_config)
    fit(model, train_loader, val_loader, training_config, LossConfig())


if __name__ == "__main__":
    main()
