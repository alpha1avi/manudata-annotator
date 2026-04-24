"""ManuData Annotator — Action Model Trainer.

Fine-tunes a video classification model (VideoMAE-v2 or TimeSformer)
on bootstrap-annotated data using standard PyTorch training with
class-weighted loss, cosine annealing, and early stopping.

Usage:
    trainer = ActionModelTrainer(config)
    trainer.train(train_dataset, val_dataset)
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MetadataFusionHead(nn.Module):
    """Small MLP that fuses video features with metadata features."""

    def __init__(self, video_dim: int, metadata_dim: int, num_classes: int) -> None:
        super().__init__()
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
        )
        self.classifier = nn.Sequential(
            nn.Linear(video_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, video_features: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        meta_out = self.metadata_mlp(metadata)
        fused = torch.cat([video_features, meta_out], dim=1)
        return self.classifier(fused)


class ActionModelTrainer:
    """Train a video classification model for action recognition."""

    SUPPORTED_MODELS = {
        "videomae-v2-small": "MCG-NJU/videomae-small-finetuned-kinetics",
        "timesformer-small": "facebook/timesformer-base-finetuned-k400",
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise trainer with configuration.

        Config keys:
            model_name: ``"videomae-v2-small"`` | ``"timesformer-small"``
            num_classes: Number of action classes.
            num_frames: Frames per clip (default 16).
            frame_size: Spatial resolution (default 224).
            lr: Learning rate (default 1e-4).
            epochs: Max training epochs (default 10).
            batch_size: Batch size (default 8).
            output_dir: Directory for checkpoints.
            use_metadata: Whether to use metadata fusion.
            metadata_dim: Dimension of metadata features (default 12).
            gpu: GPU device ID or None for auto.
        """
        self.config = config
        self.model_name = config.get("model_name", "videomae-v2-small")
        self.num_classes = config["num_classes"]
        self.num_frames = config.get("num_frames", 16)
        self.frame_size = config.get("frame_size", 224)
        self.lr = config.get("lr", 1e-4)
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 8)
        self.output_dir = config.get("output_dir", "./checkpoints")
        self.use_metadata = config.get("use_metadata", False)
        self.metadata_dim = config.get("metadata_dim", 12)

        # Device
        gpu = config.get("gpu")
        if gpu is not None:
            self.device = torch.device(f"cuda:{gpu}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info("Using device: %s", self.device)

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)

        # Fusion head for metadata (replaces the model's classifier)
        self.fusion_head: Optional[MetadataFusionHead] = None
        if self.use_metadata:
            video_dim = self._get_video_feature_dim()
            self.fusion_head = MetadataFusionHead(
                video_dim, self.metadata_dim, self.num_classes,
            )
            self.fusion_head.to(self.device)

        # Training state
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        self.best_val_accuracy: float = 0.0
        self.best_epoch: int = 0
        self.training_history: List[Dict[str, float]] = []

        os.makedirs(self.output_dir, exist_ok=True)

    # ── model loading ───────────────────────────────────────────────────

    def _load_model(self) -> nn.Module:
        """Load a pretrained model and replace the classification head."""
        hf_name = self.SUPPORTED_MODELS.get(self.model_name)
        if hf_name is None:
            raise ValueError(
                f"Unsupported model: {self.model_name}. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        logger.info("Loading pretrained model: %s (%s)", self.model_name, hf_name)

        if "videomae" in self.model_name:
            return self._load_videomae(hf_name)
        else:
            return self._load_timesformer(hf_name)

    def _load_videomae(self, hf_name: str) -> nn.Module:
        """Load VideoMAE-v2 and replace classifier head."""
        from transformers import VideoMAEForVideoClassification

        model = VideoMAEForVideoClassification.from_pretrained(
            hf_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )
        return model

    def _load_timesformer(self, hf_name: str) -> nn.Module:
        """Load TimeSformer and replace classifier head."""
        from transformers import TimesformerForVideoClassification

        model = TimesformerForVideoClassification.from_pretrained(
            hf_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )
        return model

    def _get_video_feature_dim(self) -> int:
        """Get the hidden dimension of the video encoder."""
        if hasattr(self.model, "config"):
            return getattr(self.model.config, "hidden_size", 768)
        return 768

    # ── training ────────────────────────────────────────────────────────

    def train(
        self,
        train_dataset,
        val_dataset,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            class_weights: Optional per-class weights for loss (auto-computed if None).

        Returns:
            Dict with training history and best metrics.
        """
        logger.info(
            "Starting training: %d train, %d val, %d epochs, lr=%.2e",
            len(train_dataset), len(val_dataset), self.epochs, self.lr,
        )

        if len(train_dataset) < 10:
            logger.warning(
                "Very small training set (%d samples). Results may be poor.",
                len(train_dataset),
            )

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=self.device.type == "cuda",
            drop_last=len(train_dataset) > self.batch_size,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=self.device.type == "cuda",
        )

        # Class-weighted loss
        if class_weights is None:
            class_weights = self._compute_class_weights(train_dataset)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
        )

        # Optimiser + scheduler
        params = list(self.model.parameters())
        if self.fusion_head is not None:
            params += list(self.fusion_head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs,
        )

        # Early stopping
        patience = 5
        no_improve_count = 0
        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # Train
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self._validate_epoch(val_loader)

            self.scheduler.step()
            elapsed = time.time() - t0

            epoch_stats = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": self.optimizer.param_groups[0]["lr"],
                "time_s": round(elapsed, 1),
            }
            self.training_history.append(epoch_stats)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f train_acc=%.3f "
                "val_loss=%.4f val_acc=%.3f (%.1fs)",
                epoch, self.epochs,
                train_loss, train_acc, val_loss, val_acc, elapsed,
            )

            # Best checkpoint
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(
                    os.path.join(self.output_dir, "best.pt"),
                    epoch, val_acc,
                )
                logger.info("  → New best model saved (val_acc=%.4f)", val_acc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                    break

        # Save final model and config
        self.save_checkpoint(
            os.path.join(self.output_dir, "final.pt"),
            epoch, val_acc,
        )
        self._save_training_config()

        return {
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "total_epochs": epoch,
            "history": self.training_history,
        }

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        if self.fusion_head is not None:
            self.fusion_head.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            if self.use_metadata and len(batch) == 3:
                videos, metadata, labels = batch
                metadata = metadata.to(self.device)
            else:
                videos, labels = batch[0], batch[-1]
                metadata = None

            videos = videos.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self._forward(videos, metadata)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return round(avg_loss, 4), round(accuracy, 4)

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run one validation epoch."""
        self.model.eval()
        if self.fusion_head is not None:
            self.fusion_head.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            if self.use_metadata and len(batch) == 3:
                videos, metadata, labels = batch
                metadata = metadata.to(self.device)
            else:
                videos, labels = batch[0], batch[-1]
                metadata = None

            videos = videos.to(self.device)
            labels = labels.to(self.device)

            logits = self._forward(videos, metadata)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return round(avg_loss, 4), round(accuracy, 4)

    def _forward(
        self, videos: torch.Tensor, metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through model (with optional metadata fusion)."""
        if self.fusion_head is not None and metadata is not None:
            # Get video features (before classification head)
            outputs = self.model(videos, output_hidden_states=True)
            # Use the CLS token or pooled output
            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                video_features = outputs.hidden_states[-1][:, 0, :]
            else:
                video_features = outputs.logits  # fallback
            return self.fusion_head(video_features, metadata)
        else:
            outputs = self.model(videos)
            return outputs.logits

    @staticmethod
    def _compute_class_weights(dataset) -> Optional[torch.Tensor]:
        """Compute inverse-frequency class weights for balanced loss."""
        if not hasattr(dataset, "get_class_distribution"):
            return None

        dist = dataset.get_class_distribution()
        if not dist:
            return None

        counts = []
        for cls in sorted(dataset.class_to_idx.keys()):
            counts.append(dist.get(cls, 0))

        counts = np.array(counts, dtype=np.float32)
        total = counts.sum()
        if total == 0:
            return None

        # Inverse frequency, normalised
        weights = total / (len(counts) * np.maximum(counts, 1))
        weights = weights / weights.sum() * len(counts)  # scale to mean=1

        logger.info("Class weights: %s", dict(zip(sorted(dataset.class_to_idx.keys()), weights.tolist())))
        return torch.from_numpy(weights)

    # ── checkpoint management ───────────────────────────────────────────

    def save_checkpoint(self, path: str, epoch: int, val_accuracy: float) -> None:
        """Save model state dict + optimizer + epoch + accuracy."""
        state = {
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.fusion_head is not None:
            state["fusion_head_state_dict"] = self.fusion_head.state_dict()

        torch.save(state, path)
        logger.info("Checkpoint saved: %s (epoch=%d, val_acc=%.4f)", path, epoch, val_accuracy)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load and resume training from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "fusion_head_state_dict" in checkpoint and self.fusion_head is not None:
            self.fusion_head.load_state_dict(checkpoint["fusion_head_state_dict"])

        self.best_val_accuracy = checkpoint.get("val_accuracy", 0.0)
        self.best_epoch = checkpoint.get("epoch", 0)

        logger.info(
            "Checkpoint loaded: %s (epoch=%d, val_acc=%.4f)",
            path, self.best_epoch, self.best_val_accuracy,
        )
        return checkpoint

    def _save_training_config(self) -> None:
        """Save training configuration and history."""
        config_path = os.path.join(self.output_dir, "training_config.json")
        data = {
            "config": {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                       for k, v in self.config.items()},
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "history": self.training_history,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Training config saved to %s", config_path)


# ── standalone test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)

    print("ActionModelTrainer — configuration test")
    print(f"Supported models: {list(ActionModelTrainer.SUPPORTED_MODELS.keys())}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Test MetadataFusionHead
    head = MetadataFusionHead(video_dim=768, metadata_dim=12, num_classes=5)
    v_feat = torch.randn(2, 768)
    m_feat = torch.randn(2, 12)
    out = head(v_feat, m_feat)
    print(f"Fusion head output shape: {out.shape}")  # (2, 5)
    assert out.shape == (2, 5), "Fusion head shape mismatch"

    print("\nTrainer configuration test PASSED")
    print("(Full training test requires model download — skipped in standalone mode)")
