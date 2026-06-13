"""Run CIFAR-10 ablation experiments required by Project 2."""
import copy
import csv
from pathlib import Path

from train_cifar import build_parser, run_experiment


def experiment_suite(args):
    suites = []

    if args.suite in {"filters", "all"}:
        suites.extend([
            ("filters_small", {"channels": "32,64,128", "model": "cifar_cnn"}),
            ("filters_medium", {"channels": "64,128,256", "model": "cifar_cnn"}),
            ("filters_large", {"channels": "64,128,256,512", "model": "cifar_cnn"}),
        ])

    if args.suite in {"losses", "all"}:
        suites.extend([
            ("loss_ce_no_decay", {"loss": "ce", "weight_decay": 0.0, "label_smoothing": 0.0}),
            ("loss_ce_weight_decay", {"loss": "ce", "weight_decay": 5e-4, "label_smoothing": 0.0}),
            ("loss_ce_label_smoothing", {"loss": "ce", "weight_decay": 5e-4, "label_smoothing": 0.1}),
            ("loss_focal", {"loss": "focal", "weight_decay": 5e-4, "label_smoothing": 0.0}),
        ])

    if args.suite in {"activations", "all"}:
        suites.extend([
            ("activation_relu", {"activation": "relu"}),
            ("activation_leaky_relu", {"activation": "leaky_relu"}),
            ("activation_elu", {"activation": "elu"}),
        ])

    if args.suite in {"optimizers", "all"}:
        suites.extend([
            ("optimizer_sgd", {"optimizer": "sgd", "lr": 0.05, "scheduler": "cosine", "nesterov": True}),
            ("optimizer_adam", {"optimizer": "adam", "lr": 1e-3, "scheduler": "cosine"}),
            ("optimizer_adamw", {"optimizer": "adamw", "lr": 1e-3, "scheduler": "cosine"}),
        ])

    if args.suite in {"components", "all"}:
        suites.extend([
            ("component_no_bn", {"model": "cifar_cnn", "batch_norm": False, "dropout": 0.0}),
            ("component_bn_dropout", {"model": "cifar_cnn", "batch_norm": True, "dropout": 0.2}),
            ("component_residual", {"model": "residual_cnn", "batch_norm": True, "dropout": 0.1}),
        ])

    if args.suite in {"vgg_bn", "all"}:
        suites.extend([
            ("vgg_a", {"model": "vgg_a", "optimizer": "adam", "lr": 1e-3, "scheduler": "none"}),
            ("vgg_a_batchnorm", {"model": "vgg_bn", "optimizer": "adam", "lr": 1e-3, "scheduler": "none"}),
        ])

    return suites


def clone_args(args, overrides, out_dir):
    cloned = copy.deepcopy(args)
    for key, value in overrides.items():
        setattr(cloned, key, value)
    cloned.out_dir = out_dir
    return cloned


def write_summary(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = build_parser()
    parser.description = "Run pre-defined Project 2 ablation suites."
    parser.add_argument(
        "--suite",
        choices=["filters", "losses", "activations", "optimizers", "components", "vgg_bn", "all"],
        default="all",
    )
    parser.add_argument("--max-runs", type=int, default=0, help="Debug option; 0 means run the whole suite.")
    args = parser.parse_args()

    root_out = Path(args.out_dir)
    if root_out.name == "cifar_default":
        root_out = Path("reports/experiments/ablation")
    root_out.mkdir(parents=True, exist_ok=True)

    suite = experiment_suite(args)
    if args.max_runs > 0:
        suite = suite[:args.max_runs]

    rows = []
    for index, (name, overrides) in enumerate(suite, start=1):
        run_dir = root_out / f"{index:02d}_{name}"
        run_args = clone_args(args, overrides, run_dir)
        print(f"\n=== Running {name} -> {run_dir} ===")
        result = run_experiment(run_args)
        row = {
            "name": name,
            "out_dir": result["out_dir"],
            "parameters": result["parameters"],
            "elapsed_seconds": result["elapsed_seconds"],
            "best_epoch": result["best"]["epoch"],
            "best_test_accuracy": result["best"]["val_acc"],
            "best_test_error": 1.0 - result["best"]["val_acc"],
            "model": run_args.model,
            "channels": run_args.channels,
            "activation": run_args.activation,
            "loss": run_args.loss,
            "optimizer": run_args.optimizer,
            "weight_decay": run_args.weight_decay,
            "label_smoothing": run_args.label_smoothing,
            "batch_norm": run_args.batch_norm,
            "dropout": run_args.dropout,
        }
        rows.append(row)
        write_summary(root_out / "ablation_summary.csv", rows)


if __name__ == "__main__":
    main()
