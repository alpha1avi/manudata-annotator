"""ManuData Annotator — Model Evaluator.

Evaluates a trained action recognition model on a validation set,
computing accuracy, F1, confusion matrix, and confused class pairs
that trigger VLM fallback in production mode.

Usage:
    evaluator = ModelEvaluator(model, class_mapping, device)
    result = evaluator.evaluate(val_dataset)
    evaluator.plot_confusion_matrix("confusion.png")
    evaluator.export_confused_pairs("confused_pairs.json")
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ConfusedPair:
    """A pair of classes with high mutual misclassification rate."""
    class_a: str
    class_b: str
    confusion_rate: float  # fraction of class_a predicted as class_b


@dataclass
class EvalResult:
    """Evaluation result for a trained model."""
    overall_accuracy: float
    per_class_accuracy: Dict[str, float]
    macro_f1: float
    weighted_f1: float
    confusion_matrix: np.ndarray
    class_names: List[str]
    confused_pairs: List[ConfusedPair]
    total_samples: int
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)


class ModelEvaluator:
    """Evaluate a trained action recognition model."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_mapping: Dict[str, int],
        device: torch.device,
        confusion_threshold: float = 0.10,
    ) -> None:
        """Initialise evaluator.

        Args:
            model: Trained model in eval mode.
            class_mapping: ``{class_name: class_index}`` mapping.
            device: torch device.
            confusion_threshold: Misclassification rate above which a pair
                is flagged as "confused" (default 10%).
        """
        self.model = model
        self.model.eval()
        self.class_mapping = class_mapping
        self.idx_to_class = {v: k for k, v in class_mapping.items()}
        self.class_names = [
            self.idx_to_class[i]
            for i in range(len(class_mapping))
        ]
        self.device = device
        self.confusion_threshold = confusion_threshold
        self.result: Optional[EvalResult] = None

    # ── evaluation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        val_dataset,
        batch_size: int = 8,
        use_metadata: bool = False,
    ) -> EvalResult:
        """Run model on validation set and compute metrics.

        Args:
            val_dataset: Validation dataset.
            batch_size: Batch size for evaluation.
            use_metadata: Whether dataset returns metadata.

        Returns:
            :class:`EvalResult` with all metrics.
        """
        loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
        )

        all_preds: List[int] = []
        all_labels: List[int] = []

        for batch in loader:
            if use_metadata and len(batch) == 3:
                videos, metadata, labels = batch
                videos = videos.to(self.device)
                metadata = metadata.to(self.device)
                outputs = self.model(videos, metadata)
            else:
                videos, labels = batch[0], batch[-1]
                videos = videos.to(self.device)
                outputs = self.model(videos)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

        all_preds_arr = np.array(all_preds)
        all_labels_arr = np.array(all_labels)
        num_classes = len(self.class_names)

        # Confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true, pred in zip(all_labels_arr, all_preds_arr):
            cm[true, pred] += 1

        # Overall accuracy
        overall_acc = float(np.sum(all_preds_arr == all_labels_arr)) / max(len(all_labels), 1)

        # Per-class metrics
        per_class_acc: Dict[str, float] = {}
        per_class_prec: Dict[str, float] = {}
        per_class_rec: Dict[str, float] = {}
        per_class_f1: List[float] = []
        per_class_f1_weighted: List[Tuple[float, int]] = []

        for i, cls_name in enumerate(self.class_names):
            tp = cm[i, i]
            total_true = cm[i, :].sum()
            total_pred = cm[:, i].sum()

            acc = float(tp) / max(int(total_true), 1)
            precision = float(tp) / max(int(total_pred), 1)
            recall = float(tp) / max(int(total_true), 1)

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            per_class_acc[cls_name] = round(acc, 4)
            per_class_prec[cls_name] = round(precision, 4)
            per_class_rec[cls_name] = round(recall, 4)
            per_class_f1.append(f1)
            per_class_f1_weighted.append((f1, int(total_true)))

        # Macro F1 (unweighted mean)
        macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

        # Weighted F1 (weighted by class support)
        total_support = sum(w for _, w in per_class_f1_weighted)
        if total_support > 0:
            weighted_f1 = sum(f * w for f, w in per_class_f1_weighted) / total_support
        else:
            weighted_f1 = 0.0

        # Confused pairs
        confused_pairs = self._find_confused_pairs(cm)

        self.result = EvalResult(
            overall_accuracy=round(overall_acc, 4),
            per_class_accuracy=per_class_acc,
            macro_f1=round(macro_f1, 4),
            weighted_f1=round(weighted_f1, 4),
            confusion_matrix=cm,
            class_names=self.class_names,
            confused_pairs=confused_pairs,
            total_samples=len(all_labels),
            per_class_precision=per_class_prec,
            per_class_recall=per_class_rec,
        )

        logger.info(
            "Evaluation: acc=%.3f, macro_F1=%.3f, weighted_F1=%.3f, "
            "%d confused pairs, %d samples",
            overall_acc, macro_f1, weighted_f1,
            len(confused_pairs), len(all_labels),
        )

        return self.result

    def _find_confused_pairs(self, cm: np.ndarray) -> List[ConfusedPair]:
        """Find class pairs where misclassification rate > threshold."""
        pairs: List[ConfusedPair] = []
        num_classes = cm.shape[0]

        for i in range(num_classes):
            row_total = cm[i, :].sum()
            if row_total == 0:
                continue
            for j in range(num_classes):
                if i == j:
                    continue
                rate = float(cm[i, j]) / float(row_total)
                if rate > self.confusion_threshold:
                    pairs.append(ConfusedPair(
                        class_a=self.class_names[i],
                        class_b=self.class_names[j],
                        confusion_rate=round(rate, 4),
                    ))

        # Sort by confusion rate descending
        pairs.sort(key=lambda p: p.confusion_rate, reverse=True)
        return pairs

    # ── visualisation ───────────────────────────────────────────────────

    def plot_confusion_matrix(self, save_path: str) -> None:
        """Plot and save confusion matrix using seaborn heatmap."""
        if self.result is None:
            raise RuntimeError("Call evaluate() before plotting.")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = self.result.confusion_matrix
        class_names = self.result.class_names

        # Normalise rows for display
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm.astype(np.float64),
            row_sums,
            where=row_sums > 0,
            out=np.zeros_like(cm, dtype=np.float64),
        )

        fig_size = max(8, len(class_names) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (normalised)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info("Confusion matrix saved to %s", save_path)

    # ── export ──────────────────────────────────────────────────────────

    def export_confused_pairs(self, save_path: str) -> None:
        """Save confused_pairs.json for VLM fallback triggers.

        Format::

            [
              {"class_a": "tightening_bolt", "class_b": "loosening_bolt",
               "confusion_rate": 0.23},
              ...
            ]
        """
        if self.result is None:
            raise RuntimeError("Call evaluate() before exporting.")

        pairs_data = [
            {
                "class_a": p.class_a,
                "class_b": p.class_b,
                "confusion_rate": p.confusion_rate,
            }
            for p in self.result.confused_pairs
        ]

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(pairs_data, f, indent=2)

        logger.info(
            "Confused pairs exported to %s (%d pairs)",
            save_path, len(pairs_data),
        )

    # ── reporting ───────────────────────────────────────────────────────

    def print_report(self) -> None:
        """Print formatted evaluation report."""
        if self.result is None:
            print("No evaluation results. Call evaluate() first.")
            return

        r = self.result

        print("\n" + "=" * 60)
        print("  MODEL EVALUATION REPORT")
        print("=" * 60)
        print(f"  Total samples:       {r.total_samples}")
        print(f"  Overall accuracy:    {r.overall_accuracy:.4f}")
        print(f"  Macro F1:            {r.macro_f1:.4f}")
        print(f"  Weighted F1:         {r.weighted_f1:.4f}")
        print(f"  Confused pairs:      {len(r.confused_pairs)}")
        print()

        # Per-class table
        print("  Per-class results:")
        print(f"  {'Class':<25s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s}")
        print("  " + "-" * 45)
        for cls_name in r.class_names:
            acc = r.per_class_accuracy.get(cls_name, 0.0)
            prec = r.per_class_precision.get(cls_name, 0.0)
            rec = r.per_class_recall.get(cls_name, 0.0)
            print(f"  {cls_name:<25s} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f}")

        if r.confused_pairs:
            print()
            print("  Confused pairs (trigger VLM fallback):")
            for p in r.confused_pairs[:10]:
                print(f"    {p.class_a} ↔ {p.class_b}: {p.confusion_rate:.1%}")

        print("=" * 60)


# ── standalone test ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_config import setup_logging

    setup_logging(verbose=True)

    # Synthetic test with a dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 16 * 224 * 224, num_classes)
            self.num_classes = num_classes

        def forward(self, x):
            b = x.shape[0]
            x = x.reshape(b, -1)
            # Truncate or pad to match fc input
            expected = 3 * 16 * 224 * 224
            if x.shape[1] > expected:
                x = x[:, :expected]
            elif x.shape[1] < expected:
                pad = torch.zeros(b, expected - x.shape[1])
                x = torch.cat([x, pad], dim=1)
            return self.fc(x)

    # Create synthetic dataset
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, n: int, num_classes: int):
            self.n = n
            self.num_classes = num_classes

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            video = torch.randn(3, 16, 32, 32)  # small for speed
            label = idx % self.num_classes
            return video, label

    class_mapping = {"pick_up": 0, "tighten": 1, "place": 2}
    device = torch.device("cpu")
    model = DummyModel(num_classes=3)

    evaluator = ModelEvaluator(model, class_mapping, device)
    dataset = SyntheticDataset(30, 3)

    result = evaluator.evaluate(dataset)
    evaluator.print_report()

    print(f"\nOverall accuracy: {result.overall_accuracy:.3f}")
    print(f"Confused pairs: {len(result.confused_pairs)}")
    print("Evaluation test PASSED")
