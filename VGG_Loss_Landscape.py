"""Compare VGG-A with and without BatchNorm on CIFAR-10.

This script reproduces the assignment's loss-landscape style experiment by
training VGG-A and VGG-A+BN with several learning rates, then plotting the
minimum and maximum per-step training loss envelopes.  It also records simple
gradient smoothness proxies for the report: final-layer gradient norm, gradient
change and consecutive-step cosine similarity.
"""
import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters
from utils.training import (
    ensure_dir,
    evaluate,
    get_device,
    save_checkpoint,
    save_history_csv,
    set_random_seeds,
    write_json,
)


def safe_lr_name(lr):
    return f"{lr:.0e}".replace("-", "m").replace("+", "")


def get_last_linear_weight(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return module.weight
    raise RuntimeError("No Linear layer found for gradient tracing.")


def make_optimizer(args, model, lr):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def write_step_trace(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_trace(model_name, model, lr, args, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(args, model, lr)
    model.to(device)
    target_weight = get_last_linear_weight(model)

    step_rows = []
    epoch_rows = []
    losses = []
    grad_norms = []
    grad_changes = []
    grad_cosines = []
    prev_grad = None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_items = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            grad = target_weight.grad.detach().flatten().cpu()
            grad_norm = grad.norm(2).item()
            if prev_grad is None:
                grad_change = 0.0
                grad_cosine = 1.0
            else:
                grad_change = (grad - prev_grad).norm(2).item()
                grad_cosine = F.cosine_similarity(grad, prev_grad, dim=0).item()
            prev_grad = grad.clone()

            optimizer.step()

            batch_size = y.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_correct += (logits.argmax(dim=1) == y).sum().item()
            epoch_items += batch_size

            row = {
                "step": global_step,
                "epoch": epoch,
                "lr": lr,
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "grad_change": grad_change,
                "grad_cosine": grad_cosine,
            }
            step_rows.append(row)
            losses.append(loss.item())
            grad_norms.append(grad_norm)
            grad_changes.append(grad_change)
            grad_cosines.append(grad_cosine)
            global_step += 1

        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": epoch_loss / epoch_items,
            "train_acc": epoch_correct / epoch_items,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_error": val_metrics["error"],
        }
        epoch_rows.append(epoch_row)
        print(
            f"{model_name} lr={lr:g} epoch={epoch:03d} "
            f"train_loss={epoch_row['train_loss']:.4f} train_acc={epoch_row['train_acc']:.4f} "
            f"test_acc={epoch_row['val_acc']:.4f}"
        )

    return {
        "step_rows": step_rows,
        "epoch_rows": epoch_rows,
        "losses": np.array(losses, dtype=np.float64),
        "grad_norms": np.array(grad_norms, dtype=np.float64),
        "grad_changes": np.array(grad_changes, dtype=np.float64),
        "grad_cosines": np.array(grad_cosines, dtype=np.float64),
        "model": model,
        "optimizer": optimizer,
    }


def envelope(arrays):
    min_len = min(len(array) for array in arrays)
    stacked = np.stack([array[:min_len] for array in arrays], axis=0)
    return stacked.min(axis=0), stacked.max(axis=0), stacked.mean(axis=0)


def plot_loss_landscape(path, traces_by_model):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"vgg_a": "#1f77b4", "vgg_bn": "#d62728"}
    labels = {"vgg_a": "VGG-A", "vgg_bn": "VGG-A + BN"}

    for model_name, traces in traces_by_model.items():
        min_curve, max_curve, mean_curve = envelope([trace["losses"] for trace in traces])
        steps = np.arange(len(mean_curve))
        color = colors[model_name]
        ax.plot(steps, mean_curve, color=color, label=f"{labels[model_name]} mean")
        ax.fill_between(steps, min_curve, max_curve, color=color, alpha=0.18, label=f"{labels[model_name]} min-max")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss Landscape Envelope Across Learning Rates")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_gradient_smoothness(path, traces_by_model):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"vgg_a": "#1f77b4", "vgg_bn": "#d62728"}
    labels = {"vgg_a": "VGG-A", "vgg_bn": "VGG-A + BN"}

    specs = [
        ("grad_norms", "Final-layer gradient norm", "L2 norm"),
        ("grad_changes", "Consecutive gradient change", "L2 distance"),
        ("grad_cosines", "Consecutive gradient cosine", "Cosine"),
    ]
    for axis, (key, title, ylabel) in zip(axes, specs):
        for model_name, traces in traces_by_model.items():
            _, _, mean_curve = envelope([trace[key] for trace in traces])
            axis.plot(np.arange(len(mean_curve)), mean_curve, color=colors[model_name], label=labels[model_name])
        axis.set_xlabel("Training step")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize(traces_by_model):
    summary = {}
    for model_name, traces in traces_by_model.items():
        rows = []
        for trace in traces:
            best = max(trace["epoch_rows"], key=lambda item: item["val_acc"])
            rows.append(best)
        summary[model_name] = {
            "best_test_accuracy": max(row["val_acc"] for row in rows),
            "best_test_error": 1.0 - max(row["val_acc"] for row in rows),
            "mean_final_loss": float(np.mean([trace["losses"][-1] for trace in traces])),
            "mean_gradient_change": float(np.mean([trace["grad_changes"].mean() for trace in traces])),
            "mean_gradient_cosine": float(np.mean([trace["grad_cosines"].mean() for trace in traces])),
        }
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train VGG-A with/without BatchNorm and plot loss landscape envelopes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/experiments/bn_loss_landscape"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=-1)
    parser.add_argument("--n-test", type=int, default=-1)
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 2e-3, 1e-4, 5e-4])
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--old-normalization", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    out_dir = ensure_dir(args.out_dir)
    device = get_device(args.device)
    write_json(out_dir / "config.json", vars(args) | {"device_resolved": str(device)})

    train_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=True,
        num_workers=args.num_workers,
        n_items=args.n_train,
        augment=False,
        download=not args.no_download,
        pin_memory=device.type == "cuda",
        cifar_stats=not args.old_normalization,
    )
    val_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.n_test,
        augment=False,
        download=not args.no_download,
        pin_memory=device.type == "cuda",
        cifar_stats=not args.old_normalization,
    )

    traces_by_model = {"vgg_a": [], "vgg_bn": []}
    model_builders = {
        "vgg_a": VGG_A,
        "vgg_bn": VGG_A_BatchNorm,
    }

    for model_name, builder in model_builders.items():
        for lr in args.lrs:
            set_random_seeds(args.seed, deterministic=True)
            model = builder()
            print(f"\n=== {model_name} lr={lr:g} params={get_number_of_parameters(model)} device={device} ===")
            trace = train_trace(model_name, model, lr, args, train_loader, val_loader, device)
            traces_by_model[model_name].append(trace)

            run_dir = ensure_dir(out_dir / model_name / safe_lr_name(lr))
            write_step_trace(run_dir / "step_trace.csv", trace["step_rows"])
            save_history_csv(run_dir / "epoch_metrics.csv", trace["epoch_rows"])
            save_checkpoint(
                run_dir / "last.pt",
                trace["model"],
                trace["optimizer"],
                args.epochs,
                trace["epoch_rows"][-1],
                vars(args) | {"model_name": model_name, "lr": lr},
            )

    plot_loss_landscape(out_dir / "loss_landscape.png", traces_by_model)
    plot_gradient_smoothness(out_dir / "gradient_smoothness.png", traces_by_model)
    summary = summarize(traces_by_model)
    write_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
