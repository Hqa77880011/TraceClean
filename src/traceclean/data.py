import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


NORMALIZATION = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "cifar100n": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "toy": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def symmetric_noise(labels, rate, classes, seed):
    if not 0 <= rate < 1:
        raise ValueError("Noise rate must be in [0, 1)")
    generator = np.random.default_rng(seed)
    observed = np.asarray(labels, dtype=np.int64).copy()
    indices = generator.choice(len(labels), size=int(len(labels) * rate), replace=False)
    offsets = generator.integers(1, classes, size=len(indices))
    observed[indices] = (observed[indices] + offsets) % classes
    return observed


def load_annotations(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as annotations:
            return {key: annotations[key].copy() for key in ("clean_label", "noisy_label")}
    if path.suffix != ".pt":
        raise ValueError("CIFAR-100N annotations must be a .pt or .npz file")
    from numpy.core.multiarray import _reconstruct

    allowed = [
        np.ndarray, np.dtype, _reconstruct,
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        np.dtypes.Int64DType, np.dtypes.Int32DType,
    ]
    with torch.serialization.safe_globals(allowed):
        annotations = torch.load(path, map_location="cpu", weights_only=True)
    return {key: np.asarray(annotations[key], dtype=np.int64) for key in ("clean_label", "noisy_label")}


def toy_images(per_class, seed):
    generator = np.random.default_rng(seed)
    labels = np.repeat(np.arange(10), per_class)
    images = generator.integers(0, 35, (len(labels), 32, 32, 3), dtype=np.uint8)
    for index, label in enumerate(labels):
        row = 3 + (label // 5) * 15
        column = 2 + (label % 5) * 6
        images[index, row:row + 10, column:column + 5, label % 3] = 230
    return images, labels


def prepare_data(dataset, root, output, noise_rate, noise_seed, annotations=None,
                 train_per_class=None, overwrite=False):
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; use --overwrite to replace it")
    if dataset == "cifar100n" and not annotations:
        raise ValueError("CIFAR-100N requires --annotations with the released human labels")
    if train_per_class is not None and train_per_class < 1:
        raise ValueError("train-per-class must be positive")
    if dataset == "toy":
        images, reference = toy_images(train_per_class or 12, noise_seed)
        test_images, test_labels = toy_images(4, noise_seed + 1)
        classes = 10
    else:
        factory = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
        training = factory(root=root, train=True, download=True)
        testing = factory(root=root, train=False, download=True)
        images, reference = training.data, np.asarray(training.targets, dtype=np.int64)
        test_images, test_labels = testing.data, np.asarray(testing.targets, dtype=np.int64)
        classes = 10 if dataset == "cifar10" else 100
    if dataset == "cifar100n":
        human = load_annotations(annotations)
        if not np.array_equal(human["clean_label"], reference):
            raise ValueError("Human-label reference order does not match torchvision CIFAR-100")
        observed = human["noisy_label"]
        if observed.shape != reference.shape or np.any((observed < 0) | (observed >= classes)):
            raise ValueError("Human labels must contain one valid fine-class label per training image")
    else:
        observed = symmetric_noise(reference, noise_rate, classes, noise_seed)
    indices = np.arange(len(reference))
    if train_per_class is not None and dataset != "toy":
        generator = np.random.default_rng(noise_seed)
        groups = [np.flatnonzero(reference == label) for label in range(classes)]
        if train_per_class > min(map(len, groups)):
            raise ValueError("train-per-class exceeds the available samples in a class")
        indices = np.sort(np.concatenate([
            generator.choice(group, train_per_class, replace=False) for group in groups
        ]))
        images, reference, observed = images[indices], reference[indices], observed[indices]
    metadata = {
        "dataset": dataset,
        "classes": classes,
        "noise_type": "human" if dataset == "cifar100n" else "symmetric-excluding-original",
        "requested_noise_rate": None if dataset == "cifar100n" else noise_rate,
        "actual_noise_rate": float(np.mean(observed != reference)),
        "noise_seed": noise_seed,
        "train_samples": len(reference),
        "test_samples": len(test_labels),
        "train_per_class": train_per_class,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, train_images=images, observed_labels=observed,
        reference_labels=reference, original_indices=indices,
        test_images=test_images, test_labels=test_labels,
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps({"output": str(output), **metadata}, indent=2))


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        metadata = json.loads(str(archive["metadata"].item()))
    if metadata["dataset"] not in NORMALIZATION or metadata["classes"] < 2:
        raise ValueError("Unsupported dataset metadata")
    for image_key, label_key in (("train_images", "observed_labels"), ("test_images", "test_labels")):
        images, labels = arrays[image_key], arrays[label_key]
        if images.dtype != np.uint8 or images.shape[1:] != (32, 32, 3):
            raise ValueError(f"{image_key} must be uint8 images with shape [N, 32, 32, 3]")
        if labels.shape != (len(images),) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{label_key} must be an integer label vector aligned with images")
        if len(labels) == 0 or np.any((labels < 0) | (labels >= metadata["classes"])):
            raise ValueError(f"Invalid class IDs in {label_key}")
    reference = arrays["reference_labels"]
    if reference.shape != arrays["observed_labels"].shape:
        raise ValueError("Reference labels must align with training images")
    if not np.issubdtype(reference.dtype, np.integer) or np.any((reference < 0) | (reference >= metadata["classes"])):
        raise ValueError("Invalid reference labels")
    if arrays["original_indices"].shape != reference.shape:
        raise ValueError("Original indices must align with training images")
    return arrays, metadata


class IndexedImages(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.transform(Image.fromarray(self.images[index]))
        return image, int(self.labels[index]), index


def make_loaders(arrays, metadata, config):
    mean, std = NORMALIZATION[metadata["dataset"]]
    deterministic = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    augmented = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    training = IndexedImages(arrays["train_images"], arrays["observed_labels"], augmented)
    trajectory = IndexedImages(arrays["train_images"], arrays["observed_labels"], deterministic)
    testing = IndexedImages(arrays["test_images"], arrays["test_labels"], deterministic)
    options = {"batch_size": config.batch_size, "num_workers": config.workers, "pin_memory": config.device.startswith("cuda")}
    return (
        DataLoader(training, shuffle=True, **options),
        DataLoader(trajectory, shuffle=False, **options),
        DataLoader(testing, shuffle=False, **options),
    )
