import torch
from torch.nn import functional as F


def gce_loss(logits, labels, q):
    log_probability = F.log_softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
    return -torch.expm1(q * log_probability) / q


def adaptive_loss(logits, labels, weights, q, coefficient):
    ce = F.cross_entropy(logits, labels, reduction="none")
    gce = gce_loss(logits, labels, q)
    return (weights.detach() * ce + coefficient * (1 - weights.detach()) * gce).mean()


class ELRLoss:
    def __init__(self, samples, classes, beta, coefficient, device):
        self.targets = torch.zeros(samples, classes, device=device)
        self.beta = beta
        self.coefficient = coefficient

    def __call__(self, logits, labels, indices):
        probabilities = logits.softmax(1).clamp(1e-4, 1 - 1e-4)
        detached = probabilities.detach()
        detached = detached / detached.sum(1, keepdim=True)
        with torch.no_grad():
            self.targets[indices] = self.beta * self.targets[indices] + (1 - self.beta) * detached
        agreement = (self.targets[indices] * probabilities).sum(1)
        regularization = torch.log((1 - agreement).clamp_min(1e-6)).mean()
        return F.cross_entropy(logits, labels) + self.coefficient * regularization


def coteaching_losses(logits_a, logits_b, labels, remember_rate):
    losses_a = F.cross_entropy(logits_a, labels, reduction="none")
    losses_b = F.cross_entropy(logits_b, labels, reduction="none")
    count = max(1, int(remember_rate * len(labels)))
    selected_a = losses_a.detach().argsort()[:count]
    selected_b = losses_b.detach().argsort()[:count]
    return losses_a[selected_b].mean(), losses_b[selected_a].mean()
