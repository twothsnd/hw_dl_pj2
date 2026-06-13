"""Train CIFAR-10 models for Project 2.

Example:
    python train_cifar.py --model cifar_cnn --channels 64,128,256 \
        --activation relu --optimizer adamw --epochs 80 --augment
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from data.loaders import get_cifar_loader
from models.vgg import (
    CIFARConvNet,
    VGG_A,
    VGG_A_BatchNorm,
    VGG_A_Dropout,
    get_number_of_parameters,
)
from utils.training import (
    ensure_dir,
    evaluate,
    get_device,
    plot_filters,
    plot_history,
    save_checkpoint,
    save_history_csv,
    set_random_seeds,
    train_one_epoch,
    write_json,
)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def parse_channels(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(int(item) for item in value.split(",") if item.strip())


def build_model(args):
    if args.model == "vgg_a":
        return VGG_A(num_classes=args.num_classes)
    if args.model == "vgg_bn":
        return VGG_A_BatchNorm(num_classes=args.num_classes, activation=args.activation)
    if args.model == "vgg_dropout":
        return VGG_A_Dropout(num_classes=args.num_classes)
    if args.model == "cifar_cnn":
        return CIFARConvNet(
            num_classes=args.num_classes,
            channels=parse_channels(args.channels),
            activation=args.activation,
            batch_norm=args.batch_norm,
            dropout=args.dropout,
            classifier_hidden=args.classifier_hidden,
            residual=False,
        )
    if args.model == "residual_cnn":
        return CIFARConvNet(
            num_classes=args.num_classes,
            channels=parse_channels(args.channels),
            activation=args.activation,
            batch_norm=True,
            dropout=args.dropout,
            classifier_hidden=args.classifier_hidden,
            residual=True,
        )
    raise ValueError(f"Unknown model: {args.model}")


def build_optimizer(args, model):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def build_scheduler(args, optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.scheduler == "multistep":
        milestones = [int(item) for item in args.milestones.split(",") if item.strip()]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def build_criterion(args):
    if args.loss == "ce":
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.loss == "focal":
        return FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    raise ValueError(f"Unknown loss: {args.loss}")


def config_dict(args):
    cfg = vars(args).copy()
    cfg["channels"] = ",".join(str(item) for item in parse_channels(args.channels))
    return cfg


def write_model_summary(path, model, args, best_metrics, elapsed_seconds):
    text = [
        "Project 2 CIFAR-10 experiment",
        f"model: {args.model}",
        f"channels: {','.join(str(item) for item in parse_channels(args.channels))}",
        f"activation: {args.activation}",
        f"optimizer: {args.optimizer}",
        f"loss: {args.loss}",
        f"parameters: {get_number_of_parameters(model)}",
        f"best_epoch: {best_metrics['epoch']}",
        f"best_test_accuracy: {best_metrics['val_acc']:.6f}",
        f"best_test_error: {1.0 - best_metrics['val_acc']:.6f}",
        f"elapsed_seconds: {elapsed_seconds:.2f}",
    ]
    Path(path).write_text("\n".join(text) + "\n", encoding="utf-8")


def run_experiment(args):
    set_random_seeds(args.seed, deterministic=not args.fast_cudnn)
    device = get_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    write_json(out_dir / "config.json", config_dict(args))

    train_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=True,
        num_workers=args.num_workers,
        n_items=args.n_train,
        augment=args.augment,
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

    model = build_model(args).to(device)
    criterion = build_criterion(args)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    parameter_count = get_number_of_parameters(model)

    print(f"device={device}")
    print(f"model={args.model} parameters={parameter_count}")
    print(f"train_batches={len(train_loader)} test_batches={len(val_loader)}")

    history = []
    best = {"epoch": 0, "val_acc": -1.0}
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_error": val_metrics["error"],
            "parameters": parameter_count,
        }
        history.append(row)
        print(
            "epoch={epoch:03d} lr={lr:.6g} train_loss={train_loss:.4f} "
            "train_acc={train_acc:.4f} test_loss={val_loss:.4f} "
            "test_acc={val_acc:.4f}".format(**row)
        )

        if row["val_acc"] > best["val_acc"]:
            best = row.copy()
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, row, config_dict(args))

        save_history_csv(out_dir / "metrics.csv", history)
        plot_history(out_dir / "training_curve.png", history, title=args.out_dir.name if isinstance(args.out_dir, Path) else str(args.out_dir))
        if scheduler is not None:
            scheduler.step()

    elapsed = time.perf_counter() - start
    save_checkpoint(out_dir / "last.pt", model, optimizer, args.epochs, history[-1], config_dict(args))
    plot_filters(out_dir / "first_layer_filters.png", model)
    write_json(out_dir / "summary.json", {"best": best, "parameters": parameter_count, "elapsed_seconds": elapsed})
    write_model_summary(out_dir / "model_summary.txt", model, args, best, elapsed)
    return {"best": best, "parameters": parameter_count, "elapsed_seconds": elapsed, "out_dir": str(out_dir)}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 models for Neural Network and Deep Learning Project 2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=["vgg_a", "vgg_bn", "vgg_dropout", "cifar_cnn", "residual_cnn"], default="cifar_cnn")
    parser.add_argument("--channels", default="64,128,256")
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "elu", "gelu", "silu", "tanh"], default="relu")
    parser.add_argument("--batch-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classifier-hidden", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=10)

    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/experiments/cifar_default"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=-1, help="Use a subset for quick debugging; -1 means full train set.")
    parser.add_argument("--n-test", type=int, default=-1, help="Use a subset for quick debugging; -1 means full test set.")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--old-normalization", action="store_true", help="Use the starter-code Normalize((0.5,), (0.5,)) setting.")

    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw", "rmsprop"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=0.0)

    parser.add_argument("--scheduler", choices=["none", "cosine", "step", "multistep"], default="cosine")
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--milestones", default="40,60")
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fast-cudnn", action="store_true", help="Enable cuDNN benchmark for faster non-deterministic training.")
    return parser


def main():
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
