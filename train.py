"""ManuData Annotator — Training CLI.

Fine-tune a video classification model on VLM-annotated data.

Usage:
    python train.py ./output --model videomae-v2-small --epochs 10
    python train.py ./output --augment --use-metadata --gpu 0
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import random_split

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI."""
    parser = argparse.ArgumentParser(
        prog="manudata-train",
        description="ManuData Annotator — fine-tune local models on annotated data",
    )
    parser.add_argument("data_dir", help="Directory containing annotated training data")
    parser.add_argument(
        "--model", default="videomae-v2-small",
        choices=["videomae-v2-small", "timesformer-small"],
        help="Model architecture (default: videomae-v2-small)",
    )
    parser.add_argument("--output-dir", default="./checkpoints", help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames per clip (default: 16)")
    parser.add_argument("--frame-size", type=int, default=224, help="Frame size (default: 224)")
    parser.add_argument(
        "--min-confidence", type=float, default=0.6,
        help="Minimum segment confidence for training (default: 0.6)",
    )
    parser.add_argument("--augment", action="store_true", help="Enable egocentric augmentations")
    parser.add_argument("--use-metadata", action="store_true", help="Use metadata feature fusion")
    parser.add_argument(
        "--validation-split", type=float, default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument("--gpu", default=None, help="GPU device ID (default: auto)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Logging
    from utils.logging_config import setup_logging
    setup_logging(verbose=args.verbose)

    print("=" * 60)
    print("  ManuData Annotator — Training Pipeline")
    print("=" * 60)
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Model:           {args.model}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Augmentations:   {args.augment}")
    print(f"  Metadata fusion: {args.use_metadata}")
    print(f"  Val split:       {args.validation_split}")
    print(f"  Output dir:      {args.output_dir}")
    print()

    # ── 1. Load dataset ────────────────────────────────────────────────
    from training.dataset import ActionVideoDataset

    dataset = ActionVideoDataset(
        data_dir=args.data_dir,
        min_confidence=args.min_confidence,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        augment=args.augment,
        use_metadata=args.use_metadata,
    )

    if len(dataset) == 0:
        print("ERROR: No valid segments found. Check data_dir and min-confidence.")
        sys.exit(1)

    if len(dataset) < 100:
        print(f"WARNING: Only {len(dataset)} segments found. "
              "Results may be unreliable with < 100 samples.")

    print(f"Dataset: {len(dataset)} segments, {dataset.num_classes} classes")
    dist = dataset.get_class_distribution()
    for cls_name, count in sorted(dist.items()):
        print(f"  {cls_name}: {count}")
    print()

    # ── 2. Train/val split ─────────────────────────────────────────────
    val_size = max(1, int(len(dataset) * args.validation_split))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator,
    )
    print(f"Split: {train_size} train, {val_size} validation")

    # ── 3. Create trainer ──────────────────────────────────────────────
    from training.trainer import ActionModelTrainer

    config = {
        "model_name": args.model,
        "num_classes": dataset.num_classes,
        "num_frames": args.num_frames,
        "frame_size": args.frame_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "output_dir": args.output_dir,
        "use_metadata": args.use_metadata,
        "metadata_dim": dataset.metadata_dim,
        "gpu": args.gpu,
    }

    trainer = ActionModelTrainer(config)

    # ── 4. Train ───────────────────────────────────────────────────────
    print("\nStarting training...\n")
    result = trainer.train(train_dataset, val_dataset)

    print(f"\nTraining complete!")
    print(f"  Best val accuracy: {result['best_val_accuracy']:.4f} (epoch {result['best_epoch']})")
    print(f"  Total epochs run:  {result['total_epochs']}")

    # ── 5. Evaluate ────────────────────────────────────────────────────
    from training.evaluation import ModelEvaluator

    # Load best model for evaluation
    best_path = os.path.join(args.output_dir, "best.pt")
    if os.path.exists(best_path):
        trainer.load_checkpoint(best_path)

    evaluator = ModelEvaluator(
        model=trainer.model,
        class_mapping=dataset.class_to_idx,
        device=trainer.device,
    )

    eval_result = evaluator.evaluate(
        val_dataset,
        batch_size=args.batch_size,
        use_metadata=args.use_metadata,
    )
    evaluator.print_report()

    # ── 6. Save outputs ────────────────────────────────────────────────
    # Class mapping
    dataset.save_class_mapping(os.path.join(args.output_dir, "class_mapping.json"))

    # Confusion matrix plot
    try:
        evaluator.plot_confusion_matrix(
            os.path.join(args.output_dir, "confusion_matrix.png"),
        )
    except Exception as exc:
        logger.warning("Could not save confusion matrix plot: %s", exc)

    # Confused pairs for VLM fallback
    evaluator.export_confused_pairs(
        os.path.join(args.output_dir, "confused_pairs.json"),
    )

    # ── 7. Final summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Model:            {args.model}")
    print(f"  Classes:          {dataset.num_classes}")
    print(f"  Training samples: {train_size}")
    print(f"  Val samples:      {val_size}")
    print(f"  Best val acc:     {result['best_val_accuracy']:.4f}")
    print(f"  Macro F1:         {eval_result.macro_f1:.4f}")
    print(f"  Weighted F1:      {eval_result.weighted_f1:.4f}")
    print(f"  Confused pairs:   {len(eval_result.confused_pairs)}")
    print()
    print(f"  Outputs saved to: {os.path.abspath(args.output_dir)}")
    print(f"    best.pt              — best model checkpoint")
    print(f"    final.pt             — final model checkpoint")
    print(f"    class_mapping.json   — class ↔ index mapping")
    print(f"    confused_pairs.json  — pairs for VLM fallback")
    print(f"    confusion_matrix.png — confusion matrix plot")
    print(f"    training_config.json — full training config + history")
    print("=" * 60)


if __name__ == "__main__":
    main()
