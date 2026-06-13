"""Training and evaluation helpers shared by the experiment scripts."""
import csv
import json
import os
import random
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_random_seeds(seed_value=0, deterministic=True):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(device="auto"):
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=str)


def accuracy(logits, targets):
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct / targets.size(0)


def _grad_norm(model):
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.detach().data.norm(2).item() ** 2
    return total ** 0.5


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=None, record_steps=False):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    step_losses = []
    step_grad_norms = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norm = _grad_norm(model) if record_steps else None
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_items += batch_size

        if record_steps:
            step_losses.append(loss.item())
            step_grad_norms.append(grad_norm)

    metrics = {
        "loss": total_loss / total_items,
        "accuracy": total_correct / total_items,
    }
    if record_steps:
        metrics["step_losses"] = step_losses
        metrics["step_grad_norms"] = step_grad_norms
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_items += batch_size

    return {
        "loss": total_loss / total_items,
        "accuracy": total_correct / total_items,
        "error": 1.0 - total_correct / total_items,
    }


def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def save_history_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_history(path, rows, title="Training History"):
    if not rows:
        return
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in rows], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in rows], label="test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_filters(path, model, max_filters=32):
    """Visualize first-layer convolution filters."""
    first_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            break
    if first_conv is None:
        return

    weights = first_conv.weight.detach().cpu()
    weights = weights[:max_filters]
    weights = weights - weights.amin(dim=(1, 2, 3), keepdim=True)
    weights = weights / weights.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)

    columns = min(8, len(weights))
    rows = int(np.ceil(len(weights) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.2, rows * 1.2))
    axes = np.atleast_1d(axes).reshape(rows, columns)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < len(weights):
            filt = weights[idx]
            if filt.shape[0] == 3:
                ax.imshow(np.transpose(filt.numpy(), (1, 2, 0)))
            else:
                ax.imshow(filt[0].numpy(), cmap="gray")
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)
