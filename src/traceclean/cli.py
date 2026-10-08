import argparse
from dataclasses import replace
from pathlib import Path

from .config import METHODS, load_config


ABLATIONS = ("current-loss", "ema-loss", "trajectory", "global-gmm", "class-gmm", "full",
             "class-normalize-only", "class-fit-only")


def build_parser():
    parser = argparse.ArgumentParser(prog="traceclean", description="TraceClean implementation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Download CIFAR and fix observed labels")
    prepare.add_argument("--dataset", choices=("cifar10", "cifar100", "cifar100n", "toy"), required=True)
    prepare.add_argument("--root", default="data/raw", help="torchvision download directory")
    prepare.add_argument("--output", required=True, help="Prepared .npz archive")
    prepare.add_argument("--noise-rate", type=float, default=0.4, help="Fraction of synthetic labels changed to a different class")
    prepare.add_argument("--noise-seed", type=int, default=0)
    prepare.add_argument("--annotations", help="CIFAR-100N .pt or .npz label file")
    prepare.add_argument("--train-per-class", type=int, help="Optional small training subset; toy defaults to 12")
    prepare.add_argument("--overwrite", action="store_true")
    training = commands.add_parser("train", help="Train a classifier or baseline")
    training.add_argument("--config", required=True)
    training.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override YAML settings")
    training.add_argument("--resume", help="Continue the matching last.pt checkpoint in its original run directory")
    evaluate = commands.add_parser("evaluate", help="Evaluate a saved checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", help="Override the prepared data location")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--threshold", type=float, help="Clean selection threshold")
    evaluate.add_argument("--output", help="Write evaluation JSON")
    sweep = commands.add_parser("sweep", help="Run baseline or ablation comparisons sequentially")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--suite", choices=("baselines", "ablations", "all"), required=True)
    sweep.add_argument("--seeds", nargs="+", type=int, default=[0])
    sweep.add_argument("--output", required=True, help="Parent directory for individual runs")
    sweep.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    plot = commands.add_parser("plot", help="Plot training, detection and class diagnostics")
    plot.add_argument("--run", required=True)
    plot.add_argument("--output")
    summarize = commands.add_parser("summarize", help="Aggregate actual final-epoch results across seeds")
    summarize.add_argument("--root", required=True)
    summarize.add_argument("--output", required=True)
    return parser


def main():
    arguments = build_parser().parse_args()
    if arguments.command == "prepare":
        from .data import prepare_data

        if Path(arguments.output).suffix != ".npz":
            raise ValueError("Prepared data output must end in .npz")
        if arguments.noise_seed < 0:
            raise ValueError("noise-seed must be nonnegative")
        prepare_data(arguments.dataset, arguments.root, arguments.output, arguments.noise_rate,
                     arguments.noise_seed, arguments.annotations, arguments.train_per_class, arguments.overwrite)
    elif arguments.command == "train":
        from .train import run_training

        run_training(load_config(arguments.config, arguments.set), arguments.resume)
    elif arguments.command == "evaluate":
        from .train import evaluate_checkpoint

        evaluate_checkpoint(arguments.checkpoint, arguments.data, arguments.device, arguments.output, arguments.threshold)
    elif arguments.command == "sweep":
        from .train import run_training

        config = load_config(arguments.config, arguments.set)
        combinations = []
        if arguments.suite in ("baselines", "all"):
            combinations.extend((method, "full") for method in METHODS)
        if arguments.suite in ("ablations", "all"):
            combinations.extend(("traceclean", variant) for variant in ABLATIONS)
        combinations = list(dict.fromkeys(combinations))
        if len(set(arguments.seeds)) != len(arguments.seeds):
            raise ValueError("Seeds must be distinct")
        for seed in arguments.seeds:
            for method, variant in combinations:
                name = variant if method == "traceclean" else method
                output = str(Path(arguments.output) / f"{name}-seed{seed}")
                print(f"Running {method}/{variant}, seed {seed}", flush=True)
                run_training(replace(config, method=method, variant=variant, seed=seed, output=output).validate())
    elif arguments.command == "plot":
        from .report import plot_run

        plot_run(arguments.run, arguments.output)
    elif arguments.command == "summarize":
        from .report import summarize_runs

        summarize_runs(arguments.root, arguments.output)
