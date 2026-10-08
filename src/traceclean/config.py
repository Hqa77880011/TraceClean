from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


METHODS = ("traceclean", "ce", "label-smoothing", "gce", "co-teaching", "elr", "small-loss")
VARIANTS = {
    "full": (True, True, True, True, True),
    "current-loss": (False, False, False, False, False),
    "ema-loss": (True, False, False, False, False),
    "trajectory": (True, True, False, False, False),
    "global-gmm": (True, True, False, False, True),
    "class-gmm": (True, True, True, True, False),
    "no-ema": (False, True, True, True, True),
    "no-smoothing": (True, True, True, True, True),
    "class-normalize-only": (True, True, True, False, True),
    "class-fit-only": (True, True, False, True, True),
}


@dataclass
class Config:
    data: str = "data/cifar100-sym40.npz"
    output: str = "runs/cifar100-sym40/full-seed0"
    method: str = "traceclean"
    variant: str = "full"
    seed: int = 0
    epochs: int = 200
    batch_size: int = 128
    workers: int = 0
    device: str = "auto"
    threads: int = 4
    width: int = 64
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0005
    warmup: int = 10
    window: int = 5
    alpha: float = 0.9
    rho: float = 0.9
    q: float = 0.7
    gce_lambda: float = 0.5
    epsilon: float = 0.001
    gmm_iterations: int = 100
    gmm_tolerance: float = 0.0001
    min_component_weight: float = 0.01
    threshold: float = 0.5
    label_smoothing: float = 0.1
    elr_beta: float = 0.7
    elr_lambda: float = 3.0
    forget_rate: float = 0.4
    forget_ramp: int = 10

    def validate(self):
        if self.method not in METHODS or self.variant not in VARIANTS:
            raise ValueError("Unknown method or variant")
        if self.method != "traceclean" and self.variant != "full":
            raise ValueError("Ablation variants apply only to TraceClean")
        if self.epochs < 1 or self.warmup < 0 or self.window < 2:
            raise ValueError("Require epochs >= 1, warmup >= 0, window >= 2")
        if min(self.batch_size, self.width, self.threads, self.gmm_iterations, self.forget_ramp) < 1:
            raise ValueError("Batch size, width, threads and iteration counts must be positive")
        if self.workers < 0 or self.seed < 0:
            raise ValueError("Workers and seed must be nonnegative")
        if not 0 < self.q <= 1 or not 0 < self.threshold < 1:
            raise ValueError("Require 0 < q <= 1 and 0 < threshold < 1")
        for name in ("alpha", "rho", "elr_beta", "forget_rate", "label_smoothing"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if min(self.lr, self.epsilon, self.gmm_tolerance) <= 0:
            raise ValueError("Learning rate, epsilon and GMM tolerance must be positive")
        if min(self.gce_lambda, self.elr_lambda, self.weight_decay, self.momentum) < 0:
            raise ValueError("Loss coefficients and optimizer parameters must be nonnegative")
        if not 0 < self.min_component_weight < 0.5:
            raise ValueError("min_component_weight must be in (0, 0.5)")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path, overrides=()):
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    allowed = {field.name: field.type for field in fields(Config)}
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"Expected key=value, received {override}")
        values[key] = yaml.safe_load(value)
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for key, value in values.items():
        expected = allowed[key]
        if expected is float and isinstance(value, (float, int)) and not isinstance(value, bool):
            values[key] = float(value)
        elif not isinstance(value, expected) or isinstance(value, bool):
            raise ValueError(f"{key} must have type {expected.__name__}")
    return Config(**values).validate()
