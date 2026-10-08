import csv
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from .config import Config
from .data import load_data, make_loaders
from .losses import ELRLoss, adaptive_loss, coteaching_losses, gce_loss
from .metrics import classification_metrics, detection_metrics
from .model import PreActResNet18
from .trajectory import Trajectory


HISTORY_FIELDS = (
    "epoch", "train_loss", "test_loss", "top1", "worst_class_accuracy", "lr",
    "auroc", "clean_precision", "clean_recall", "selected_fraction", "mean_weight",
    "class_fits", "fallback_samples", "retained_samples", "train_seconds", "refresh_seconds",
)


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Device must be cpu, cuda, cuda:N or mps")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return str(resolved)


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def collect_predictions(models, loader, device):
    for model in models:
        model.eval()
    count = len(loader.dataset)
    losses = np.empty(count, dtype=np.float32)
    predictions = np.empty(count, dtype=np.int64)
    labels = np.empty(count, dtype=np.int64)
    for images, targets, indices in loader:
        images, targets = images.to(device), targets.to(device)
        probabilities = torch.stack([model(images).softmax(1) for model in models]).mean(0)
        log_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        losses[indices.numpy()] = F.nll_loss(log_probabilities, targets, reduction="none").cpu().numpy()
        predictions[indices.numpy()] = probabilities.argmax(1).cpu().numpy()
        labels[indices.numpy()] = targets.cpu().numpy()
    return losses, predictions, labels


def train_epoch(models, optimizers, loader, config, epoch, trajectory, elr):
    for model in models:
        model.train()
    total_loss = 0.0
    training_weights = trajectory.training_weights() if trajectory is not None else None
    remember = 1 - config.forget_rate * min(1.0, (epoch - 1) / config.forget_ramp)
    for images, labels, indices in loader:
        images, labels = images.to(config.device), labels.to(config.device)
        logits = [model(images) for model in models]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        if config.method == "co-teaching":
            loss_a, loss_b = coteaching_losses(logits[0], logits[1], labels, remember)
            loss_a.backward()
            loss_b.backward()
            loss = (loss_a.detach() + loss_b.detach()) / 2
        else:
            output = logits[0]
            if config.method == "traceclean":
                weights = torch.as_tensor(training_weights[indices.numpy()], device=config.device)
                loss = adaptive_loss(output, labels, weights, config.q, config.gce_lambda)
            elif config.method == "gce":
                loss = gce_loss(output, labels, config.q).mean()
            elif config.method == "label-smoothing":
                loss = F.cross_entropy(output, labels, label_smoothing=config.label_smoothing)
            elif config.method == "elr":
                loss = elr(output, labels, indices.to(config.device))
            elif config.method == "small-loss" and epoch > config.warmup:
                individual = F.cross_entropy(output, labels, reduction="none")
                keep = max(1, int((1 - config.forget_rate) * len(labels)))
                loss = individual[individual.detach().argsort()[:keep]].mean()
            else:
                loss = F.cross_entropy(output, labels)
            loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
        for optimizer in optimizers:
            optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
    return total_loss / len(loader.dataset)


def tensor_state(state):
    return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value for key, value in state.items()}


def numpy_state(state):
    return {key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in state.items()}


def save_checkpoint(path, models, optimizers, schedulers, config, metadata, epoch, trajectory, elr):
    numpy_rng = np.random.get_state()
    payload = {
        "config": config.to_dict(), "data_metadata": metadata, "epoch": epoch,
        "models": [model.state_dict() for model in models],
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "schedulers": [scheduler.state_dict() for scheduler in schedulers],
        "trajectory": tensor_state(trajectory.state_dict()) if trajectory is not None else None,
        "elr_targets": elr.targets if elr is not None else None,
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
    }
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(checkpoint, models, optimizers, schedulers, trajectory, elr):
    for model, state in zip(models, checkpoint["models"]):
        model.load_state_dict(state)
    for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
        optimizer.load_state_dict(state)
    for scheduler, state in zip(schedulers, checkpoint["schedulers"]):
        scheduler.load_state_dict(state)
    if trajectory is not None:
        trajectory.load_state_dict(numpy_state(checkpoint["trajectory"]))
    if elr is not None:
        elr.targets.copy_(checkpoint["elr_targets"])
    random.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    if torch.cuda.is_available() and checkpoint["cuda_rng"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    numpy_rng = checkpoint["numpy_rng"]
    np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))


def run_training(config, resume=None):
    config = replace(config, device=resolve_device(config.device)).validate()
    seed_everything(config.seed, config.threads)
    arrays, metadata = load_data(config.data)
    output = Path(config.output)
    existing = list(output.iterdir()) if output.exists() else []
    if existing and resume is None:
        raise FileExistsError(f"{output} is not empty; choose another output or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    models = [PreActResNet18(metadata["classes"], config.width).to(config.device)]
    if config.method == "co-teaching":
        models.append(PreActResNet18(metadata["classes"], config.width).to(config.device))
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                        weight_decay=config.weight_decay)
        for model in models
    ]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs) for optimizer in optimizers]
    trajectory = Trajectory(arrays["observed_labels"], config) if config.method == "traceclean" else None
    elr = (
        ELRLoss(len(arrays["observed_labels"]), metadata["classes"], config.elr_beta, config.elr_lambda, config.device)
        if config.method == "elr" else None
    )
    start = 1
    if resume:
        checkpoint = torch.load(resume, map_location=config.device, weights_only=True)
        saved = checkpoint["config"]
        changing = [key for key, value in config.to_dict().items() if key not in ("device", "threads", "workers") and saved[key] != value]
        if changing or checkpoint["data_metadata"] != metadata:
            raise ValueError(f"Resume requires the original data and configuration; changed keys: {changing}")
        restore_checkpoint(checkpoint, models, optimizers, schedulers, trajectory, elr)
        start = checkpoint["epoch"] + 1
        if start > config.epochs:
            raise ValueError("The checkpoint has already completed the configured epochs")
        if not (output / "history.csv").exists():
            raise FileNotFoundError("Resume requires history.csv in the original output directory")
        with (output / "history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        if not history or int(history[-1]["epoch"]) != start - 1:
            raise ValueError("history.csv and the resume checkpoint refer to different epochs")
    training_loader, trajectory_loader, testing_loader = make_loaders(arrays, metadata, config)
    (output / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    write_json(output / "data.json", metadata)
    history_path = output / "history.csv"
    if not resume:
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writeheader()
    final_predictions = None
    for epoch in range(start, config.epochs + 1):
        started = time.perf_counter()
        learning_rate = optimizers[0].param_groups[0]["lr"]
        train_loss = train_epoch(models, optimizers, training_loader, config, epoch, trajectory, elr)
        train_seconds = time.perf_counter() - started
        for scheduler in schedulers:
            scheduler.step()
        test_loss, test_predictions, test_labels = collect_predictions(models, testing_loader, config.device)
        classification = classification_metrics(test_labels, test_predictions, metadata["classes"])
        refresh_started = time.perf_counter()
        details = {"class_fits": 0, "fallback_samples": 0, "retained_samples": 0}
        detection = {}
        scores = None
        if trajectory is not None and epoch > config.warmup:
            losses, predictions, _ = collect_predictions(models, trajectory_loader, config.device)
            details = trajectory.update(losses, predictions)
            if details["window_ready"]:
                scores = trajectory.weights
                detection = detection_metrics(arrays["observed_labels"], arrays["reference_labels"], scores,
                                              metadata["classes"], config.threshold)
        elif trajectory is None and epoch == config.epochs:
            losses, predictions, _ = collect_predictions(models, trajectory_loader, config.device)
            scores = np.exp(-losses)
            detection = detection_metrics(arrays["observed_labels"], arrays["reference_labels"], scores,
                                          metadata["classes"], config.threshold)
        refresh_seconds = time.perf_counter() - refresh_started
        row = {
            "epoch": epoch, "train_loss": train_loss, "test_loss": float(test_loss.mean()),
            "top1": classification["top1"], "worst_class_accuracy": classification["worst_class_accuracy"],
            "lr": learning_rate, "train_seconds": train_seconds, "refresh_seconds": refresh_seconds,
            **{key: detection.get(key) for key in ("auroc", "clean_precision", "clean_recall", "selected_fraction")},
            **{key: details.get(key) for key in ("class_fits", "fallback_samples", "retained_samples", "mean_weight")},
        }
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writerow(row)
        save_checkpoint(output / "last.pt", models, optimizers, schedulers, config, metadata, epoch, trajectory, elr)
        print(f"epoch={epoch}/{config.epochs} loss={train_loss:.4f} top1={classification['top1']:.4f} "
              f"worst={classification['worst_class_accuracy']:.4f} fits={details['class_fits']} "
              f"fallback={details['fallback_samples']}", flush=True)
        final_predictions = (test_predictions, test_labels)
    summary = {
        "method": config.method, "variant": config.variant, "seed": config.seed, "epoch": config.epochs,
        "data": metadata, "config": config.to_dict(), "classification": classification,
        "detection": detection or None,
        "score_type": "smoothed-reliability" if trajectory is not None else "observed-label-probability",
        "trajectory_records": trajectory.records if trajectory is not None else 0,
        "first_weighted_epoch": config.warmup + config.window + 1 if trajectory is not None else None,
    }
    write_json(output / "summary.json", summary)
    np.savez_compressed(output / "test_predictions.npz", predictions=final_predictions[0], labels=final_predictions[1])
    if scores is not None:
        artifacts = {
            "scores": scores, "observed_labels": arrays["observed_labels"],
            "reference_labels": arrays["reference_labels"], "original_indices": arrays["original_indices"],
            "loss": losses, "predictions": predictions,
        }
        if trajectory is not None:
            artifacts.update(posterior=trajectory.posterior, features=trajectory.features, oriented=trajectory.oriented)
        np.savez_compressed(output / "scores.npz", **artifacts)
    print(f"Saved final-epoch metrics and checkpoint to {output}", flush=True)
    return summary


def evaluate_checkpoint(path, data=None, device="auto", output=None, threshold=None):
    resolved = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved, weights_only=True)
    values = checkpoint["config"].copy()
    values.update(device=resolved, data=data or values["data"])
    if threshold is not None:
        values["threshold"] = threshold
    config = Config(**values).validate()
    seed_everything(config.seed, config.threads)
    arrays, metadata = load_data(config.data)
    if metadata != checkpoint["data_metadata"]:
        raise ValueError("Evaluation data metadata differs from the checkpoint")
    models = []
    for state in checkpoint["models"]:
        model = PreActResNet18(metadata["classes"], config.width).to(resolved)
        model.load_state_dict(state)
        models.append(model)
    _, trajectory_loader, testing_loader = make_loaders(arrays, metadata, config)
    losses, predictions, labels = collect_predictions(models, testing_loader, resolved)
    result = {"epoch": checkpoint["epoch"], "test_loss": float(losses.mean()),
              "classification": classification_metrics(labels, predictions, metadata["classes"])}
    state = checkpoint["trajectory"]
    if state is not None and state["records"] >= config.window:
        scores = state["weights"].cpu().numpy()
    elif state is None:
        training_losses, _, _ = collect_predictions(models, trajectory_loader, resolved)
        scores = np.exp(-training_losses)
    else:
        scores = None
    result["detection"] = (
        detection_metrics(arrays["observed_labels"], arrays["reference_labels"], scores,
                          metadata["classes"], config.threshold) if scores is not None else None
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    return result
