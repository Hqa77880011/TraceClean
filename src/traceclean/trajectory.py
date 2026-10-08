import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from .config import VARIANTS


def robust_normalize(features, labels, class_relative, epsilon):
    normalized = np.empty_like(features, dtype=np.float64)
    groups = np.unique(labels) if class_relative else [None]
    for group in groups:
        indices = np.arange(len(labels)) if group is None else np.flatnonzero(labels == group)
        values = features[indices]
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        normalized[indices] = (values - median) / (mad + epsilon)
    normalized[:, 0] *= -1
    return normalized


def diagonal_mixture(values, config, seed):
    samples, dimensions = values.shape
    if samples < 2 * (dimensions + 1) or not np.isfinite(values).all():
        return None
    if len(np.unique(values, axis=0)) < 2:
        return None
    with threadpool_limits(limits=1):
        assignments = KMeans(n_clusters=2, n_init=3, random_state=seed).fit_predict(values)
    means = np.stack([values[assignments == component].mean(0) for component in range(2)])
    variances = np.stack([
        np.maximum(values[assignments == component].var(0), config.epsilon)
        for component in range(2)
    ])
    mixing = np.bincount(assignments, minlength=2) / samples
    previous = None
    converged = False
    for iteration in range(config.gmm_iterations):
        log_density = -0.5 * (
            dimensions * np.log(2 * np.pi)
            + np.log(variances).sum(1)[None, :]
            + (((values[:, None, :] - means[None, :, :]) ** 2) / variances[None, :, :]).sum(2)
        ) + np.log(mixing)[None, :]
        log_normalizer = np.logaddexp(log_density[:, 0], log_density[:, 1])
        responsibilities = np.exp(log_density - log_normalizer[:, None])
        likelihood = float(log_normalizer.mean())
        if not np.isfinite(likelihood):
            return None
        if previous is not None and abs(likelihood - previous) < config.gmm_tolerance:
            converged = True
            break
        counts = responsibilities.sum(0)
        if np.min(counts / samples) < config.min_component_weight:
            return None
        mixing = counts / samples
        means = (responsibilities.T @ values) / counts[:, None]
        difference = values[:, None, :] - means[None, :, :]
        variances = np.maximum(
            (responsibilities[:, :, None] * difference ** 2).sum(0) / counts[:, None],
            config.epsilon,
        )
        previous = likelihood
    if not converged or np.linalg.norm(means[0] - means[1]) < config.epsilon:
        return None
    reliable = int(np.argmax(means.mean(1)))
    return responsibilities[:, reliable].astype(np.float32)


class Trajectory:
    def __init__(self, labels, config):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.config = config
        self.history = np.full((len(labels), config.window), -1, dtype=np.int16)
        self.records = 0
        self.ema_loss = np.zeros(len(labels), dtype=np.float32)
        self.weights = np.ones(len(labels), dtype=np.float32)
        self.posterior = np.ones(len(labels), dtype=np.float32)
        self.features = np.zeros((len(labels), 3), dtype=np.float32)
        self.oriented = np.zeros((len(labels), 3), dtype=np.float64)

    def update(self, losses, predictions):
        config = self.config
        use_ema, use_trajectory, normalize_class, fit_class, soft = VARIANTS[config.variant]
        losses = np.asarray(losses, dtype=np.float32)
        if self.records == 0:
            self.ema_loss[:] = losses
        else:
            self.ema_loss[:] = config.alpha * self.ema_loss + (1 - config.alpha) * losses
        self.history[:, self.records % config.window] = predictions
        self.records += 1
        if self.records < config.window:
            return {"window_ready": False, "class_fits": 0, "fallback_samples": 0, "retained_samples": 0}
        order = (np.arange(config.window) + self.records % config.window) % config.window
        history = self.history[:, order]
        stability = 1 - np.mean(history[:, 1:] != history[:, :-1], axis=1)
        agreement = np.mean(history == self.labels[:, None], axis=1)
        self.features = np.column_stack((self.ema_loss if use_ema else losses, stability, agreement))
        features = self.features if use_trajectory else self.features[:, :1]
        self.oriented = robust_normalize(features, self.labels, normalize_class, config.epsilon)
        posterior = self.posterior.copy()
        successful = 0
        fallback_samples = 0
        retained_samples = 0
        global_posterior = None
        global_attempted = False
        groups = np.unique(self.labels) if fit_class else [None]
        for group in groups:
            indices = np.arange(len(self.labels)) if group is None else np.flatnonzero(self.labels == group)
            result = diagonal_mixture(self.oriented[indices], config, config.seed + self.records)
            if result is not None:
                posterior[indices] = result
                successful += int(group is not None)
                continue
            if fit_class:
                if not global_attempted:
                    global_posterior = diagonal_mixture(self.oriented, config, config.seed + self.records)
                    global_attempted = True
                if global_posterior is not None:
                    posterior[indices] = global_posterior[indices]
                    fallback_samples += len(indices)
                    continue
            retained_samples += len(indices)
        self.posterior = posterior
        rho = 0.0 if config.variant == "no-smoothing" else config.rho
        self.weights = (rho * self.weights + (1 - rho) * posterior).astype(np.float32)
        return {
            "window_ready": True,
            "class_fits": successful,
            "fallback_samples": fallback_samples,
            "retained_samples": retained_samples,
            "mean_weight": float(self.weights.mean()),
            "soft": soft,
        }

    def training_weights(self):
        if VARIANTS[self.config.variant][4]:
            return self.weights
        return (self.weights >= self.config.threshold).astype(np.float32)

    def state_dict(self):
        return {
            "history": self.history,
            "records": self.records,
            "ema_loss": self.ema_loss,
            "weights": self.weights,
            "posterior": self.posterior,
            "features": self.features,
            "oriented": self.oriented,
        }

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, value)
