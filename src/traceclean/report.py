import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from .train import write_json


COLORS = ("#447597", "#6c9891", "#8b969f")
METRICS = ("top1", "worst_class_accuracy", "auroc", "clean_precision", "clean_recall")


def numeric_column(rows, key):
    return np.array([float(row[key]) if row.get(key) not in (None, "") else np.nan for row in rows])


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_run(run, output=None):
    run = Path(run)
    output = Path(output) if output else run / "figures"
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    with (run / "history.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = numeric_column(rows, "epoch")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(epochs, numeric_column(rows, "train_loss"), label="Train objective", color=COLORS[0])
    axes[0].plot(epochs, numeric_column(rows, "test_loss"), label="Test CE", color=COLORS[1])
    axes[0].set_ylabel("Loss")
    axes[1].plot(epochs, 100 * numeric_column(rows, "top1"), label="Top-1", color=COLORS[0])
    axes[1].plot(epochs, 100 * numeric_column(rows, "worst_class_accuracy"), label="Worst class", color=COLORS[1])
    axes[1].set_ylabel("Accuracy (%)")
    for key, color, label in zip(("auroc", "clean_precision", "clean_recall"), COLORS, ("AUROC", "Clean precision", "Clean recall")):
        values = numeric_column(rows, key)
        if np.isfinite(values).any():
            axes[2].plot(epochs, 100 * values, marker="." if np.isfinite(values).sum() == 1 else None, label=label, color=color)
    axes[2].set_ylabel("Detection (%)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.15)
        if axis.lines:
            axis.legend(fontsize=8)
    save_figure(figure, output / "training.png")
    classification = summary["classification"]
    per_class = np.asarray(classification["per_class_accuracy"], dtype=float)
    detection = summary["detection"]
    figure, axes = plt.subplots(2 if detection else 1, 1, figsize=(10, 5 if detection else 3), squeeze=False)
    axes[0, 0].bar(np.arange(len(per_class)), 100 * per_class, color=COLORS[0])
    axes[0, 0].set_ylabel("Test accuracy (%)")
    axes[0, 0].set_xlabel("True test class")
    if detection:
        recalls = np.asarray(detection["per_class_clean_recall"], dtype=float)
        axes[1, 0].bar(np.arange(len(recalls)), 100 * recalls, color=COLORS[1])
        axes[1, 0].set_ylabel("Clean recall (%)")
        axes[1, 0].set_xlabel("Observed training class")
    save_figure(figure, output / "classes.png")
    score_path = run / "scores.npz"
    if score_path.exists():
        with np.load(score_path, allow_pickle=False) as archive:
            scores = archive["scores"]
            clean = archive["observed_labels"] == archive["reference_labels"]
            figure, axes = plt.subplots(1, 2, figsize=(8, 3.4))
            for mask, color, label in ((clean, COLORS[0], "Clean"), (~clean, COLORS[1], "Noisy")):
                if mask.any():
                    axes[0].hist(scores[mask], bins=np.linspace(0, 1, 31), alpha=0.65, color=color, label=label)
            axes[0].axvline(summary["config"]["threshold"], color=COLORS[2], linestyle="--")
            axes[0].set_xlabel(summary["score_type"])
            axes[0].set_ylabel("Samples")
            axes[0].legend()
            if np.unique(clean).size == 2:
                false_positive, true_positive, _ = roc_curve(clean, scores)
                axes[1].plot(false_positive, true_positive, color=COLORS[0], label=f"AUROC = {detection['auroc']:.3f}")
                axes[1].plot([0, 1], [0, 1], linestyle="--", color=COLORS[2])
                axes[1].legend()
            axes[1].set_xlabel("False positive rate")
            axes[1].set_ylabel("Clean recall")
            save_figure(figure, output / "reliability.png")
            if "features" in archive:
                features = archive["features"]
                generator = np.random.default_rng(summary["seed"])
                sample = generator.choice(len(scores), min(5000, len(scores)), replace=False)
                figure, axes = plt.subplots(1, 2, figsize=(8, 3.4))
                for axis, coordinate, label in zip(axes, (1, 2), ("Prediction stability", "Label agreement")):
                    dots = axis.scatter(features[sample, 0], features[sample, coordinate], c=scores[sample], s=5,
                                        cmap="viridis", vmin=0, vmax=1, alpha=0.6)
                    axis.set_xlabel("Loss trajectory")
                    axis.set_ylabel(label)
                    figure.colorbar(dots, ax=axis, label="Smoothed reliability")
                save_figure(figure, output / "trajectories.png")
    print(f"Saved figures to {output}")


def summarize_runs(root, output):
    root, output = Path(root), Path(output)
    paths = sorted(root.rglob("summary.json"))
    if not paths:
        raise FileNotFoundError(f"No summary.json files below {root}")
    groups = {}
    seen = set()
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        settings = summary["config"].copy()
        for key in ("seed", "output", "method", "variant", "device", "workers", "threads"):
            settings.pop(key, None)
        key = (
            json.dumps(summary["data"], sort_keys=True), json.dumps(settings, sort_keys=True),
            summary["method"], summary["variant"],
        )
        observation = (*key, summary["seed"])
        if observation in seen:
            raise ValueError(f"Duplicate method/variant/seed observation at {path}")
        seen.add(observation)
        values = {**summary["classification"], **(summary["detection"] or {})}
        groups.setdefault(key, []).append((summary, values, str(path)))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = []
    for group_id, (key, entries) in enumerate(groups.items(), start=1):
        data = entries[0][0]["data"]
        row = {
            "group": group_id, "dataset": data["dataset"], "noise_type": data["noise_type"],
            "noise_rate": data["actual_noise_rate"], "noise_seed": data["noise_seed"],
            "method": key[2], "variant": key[3], "runs": len(entries),
        }
        for metric in METRICS:
            values = [values[metric] for _, values, _ in entries if values.get(metric) is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            row[f"{metric}_n"] = len(values)
        rows.append(row)
        sources.append({"group": group_id, "data": data, "training_settings": json.loads(key[1]),
                        "seeds": [summary["seed"] for summary, _, _ in entries],
                        "files": [path for _, _, path in entries]})
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "comparison.json", {"rows": rows, "sources": sources})
    lines = ["| Dataset | Noise | Method | Variant | Seeds | Top-1 (%) | Worst class (%) | AUROC (%) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        formatted = []
        for metric in METRICS[:3]:
            mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
            text = "—" if mean is None else f"{100 * mean:.2f}"
            if std is not None:
                text += f" ± {100 * std:.2f}"
            formatted.append(text)
        lines.append(f"| {row['dataset']} | {row['noise_rate']:.3f} | {row['method']} | {row['variant']} | {row['runs']} | "
                     + " | ".join(formatted) + " |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(max(6, len(rows) * 0.75), 4))
    labels = [f"{row['dataset']}\n{row['method']}\n{row['variant']}\nnoise={row['noise_rate']:.2f}" for row in rows]
    means = [100 * row["top1_mean"] for row in rows]
    deviations = [100 * row["top1_std"] if row["top1_std"] is not None else 0 for row in rows]
    axis.bar(np.arange(len(rows)), means, yerr=deviations, capsize=3, color=COLORS[0])
    axis.set_xticks(np.arange(len(rows)), labels, fontsize=8, rotation=35, ha="right")
    axis.set_ylabel("Final-epoch test accuracy (%)")
    axis.grid(axis="y", alpha=0.15)
    save_figure(figure, output / "comparison.png")
    print(f"Summarized {len(paths)} runs into {output}")
