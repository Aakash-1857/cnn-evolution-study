import copy
import numpy as np
import matplotlib.pyplot as plt
import torch


# ── METRIC STORE ──────────────────────────────────────────────────────────────

class MetricStore:
    def __init__(self):
        self.data = {}

    def update(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def epoch_mean(self, key):
        vals = self.data.pop(key)
        return float(np.mean(vals))

    def epoch_mean_std(self, key):
        vals = self.data.pop(key)
        return float(np.mean(vals)), float(np.std(vals))


# ── GRADIENT NORM ─────────────────────────────────────────────────────────────

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().norm(2).item() ** 2
    return total_norm ** 0.5

# -- PARAMETER COUNT -----------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# ── CHECKPOINT MANAGER ────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self):
        self.best_val_acc  = -1.0
        self.best_epoch    = -1
        self.best_weights  = None

    def update(self, model, val_acc, epoch):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch   = epoch
            self.best_weights = copy.deepcopy(model.state_dict())


# ── TRAIN EPOCH ───────────────────────────────────────────────────────────────

def run_train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    store = MetricStore()

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        yhat = model(X)
        loss = loss_fn(yhat, y)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = compute_grad_norm(model)
        optimizer.step()

        acc = 100 * (torch.argmax(yhat, dim=1) == y).float().mean().item()
        store.update("train_loss", loss.item())
        store.update("train_acc",  acc)
        store.update("grad_norm",  grad_norm)

    grad_norm_mean, grad_norm_std = store.epoch_mean_std("grad_norm")
    return (
        store.epoch_mean("train_loss"),
        store.epoch_mean("train_acc"),
        grad_norm_mean,
        grad_norm_std,
    )


# ── VAL EPOCH ─────────────────────────────────────────────────────────────────

def run_val_epoch(model, val_loader, loss_fn, device):
    model.eval()
    store = MetricStore()

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            yhat = model(X)
            loss = loss_fn(yhat, y)
            acc  = 100 * (torch.argmax(yhat, dim=1) == y).float().mean().item()
            store.update("val_loss", loss.item())
            store.update("val_acc",  acc)

    model.train()
    return store.epoch_mean("val_loss"), store.epoch_mean("val_acc")


# ── OUTER LOOP ────────────────────────────────────────────────────────────────

def train_model(model, loss_fn, optimizer, train_loader, val_loader,
                epochs=10, scheduler=None, device="cpu"):
    
    config = {
        "epochs":     epochs,
        "lr":         optimizer.param_groups[0]["lr"],
        "batch_size": train_loader.batch_size,
        "optimizer":  type(optimizer).__name__,
        "scheduler":  type(scheduler).__name__ if scheduler else None,
        "parameters": count_parameters(model),
    }
    print(f"Trainable Parameters: {config['parameters']:,}")

    history = {
        "train_loss":    [], "val_loss": [],
        "train_acc":     [], "val_acc":  [],
        "grad_norms":    [], "grad_norm_std": [],
        "lr":            [],
    }
    ckpt = CheckpointManager()

    for epoch in range(epochs):
        tr_loss, tr_acc, grad_norm, grad_norm_std = run_train_epoch(
            model, train_loader, loss_fn, optimizer, device)
        
        val_loss, val_acc = run_val_epoch(
            model, val_loader, loss_fn, device)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        ckpt.update(model, val_acc, epoch)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["grad_norms"].append(grad_norm)
        history["grad_norm_std"].append(grad_norm_std)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1:02d} | "
              f"train_loss: {tr_loss:.3f} | val_loss: {val_loss:.3f} | "
              f"train_acc: {tr_acc:.1f}% | val_acc: {val_acc:.1f}% | "
              f"grad_norm: {grad_norm:.3f} ± {grad_norm_std:.3f}")

    return {
        "history":      history,
        "best_val_acc": ckpt.best_val_acc,
        "best_epoch":   ckpt.best_epoch,
        "best_weights": ckpt.best_weights,
        "config":       config,
    }


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_training_results(history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0,0].plot(history["train_loss"], label="train")
    axes[0,0].plot(history["val_loss"],   label="val")
    axes[0,0].set_title("Loss")
    axes[0,0].legend()

    axes[0,1].plot(history["train_acc"], label="train")
    axes[0,1].plot(history["val_acc"],   label="val")
    axes[0,1].set_title("Accuracy")
    axes[0,1].legend()

    epochs_range = range(len(history["grad_norms"]))
    grad_mean = np.array(history["grad_norms"])
    grad_std  = np.array(history["grad_norm_std"])
    axes[1,0].plot(epochs_range, grad_mean, label="mean")
    axes[1,0].fill_between(epochs_range,
                           grad_mean - grad_std,
                           grad_mean + grad_std,
                           alpha=0.3, label="±1 std")
    axes[1,0].set_title("Gradient Norm")
    axes[1,0].legend()

    axes[1,1].plot(history["lr"])
    axes[1,1].set_title("Learning Rate")

    plt.tight_layout()
    plt.show()

def inspect_model(history):
    """
    Plots a 4-panel diagnostic dashboard for model training:
    1. Loss Gap (Overfitting Check)
    2. Validation Accuracy Plateau
    3. Gradient Norms (Scale)
    4. Gradient Coefficient of Variation (Batch Variance)
    """
    epochs_count = len(history.get("train_loss", []))
    if epochs_count == 0:
        epochs_count = len(history.get("val_acc", []))
    epochs = np.arange(1, epochs_count + 1)

    # Initialize a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # ----------------------------------------------------
    # PANEL 1: Loss Gap (Overfitting Analysis)
    # ----------------------------------------------------
    if "train_loss" in history and "val_loss" in history:
        gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
        axs[0, 0].plot(epochs, gap, color="crimson", label="Loss Gap (Val - Train)")

        # Find first epoch where gap > 0.3
        overfit_epoch = None
        for epoch_idx, val in enumerate(gap):
            if val > 0.3:
                overfit_epoch = epoch_idx + 1
                break

        if overfit_epoch is not None:
            axs[0, 0].axvline(x=overfit_epoch, color='red', linestyle='--', label=f"Overfitting (>0.3) @ Ep {overfit_epoch}")

        axs[0, 0].set_title("Loss Gap (Overfitting Check)")
        axs[0, 0].set_xlabel("Epochs")
        axs[0, 0].set_ylabel("Gap Value")
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle=":")
    else:
        axs[0, 0].text(0.5, 0.5, "Loss data missing", ha='center', va='center')

    # ----------------------------------------------------
    # PANEL 2: Validation Accuracy (Plateau Analysis)
    # ----------------------------------------------------
    if "val_acc" in history:
        val_acc = history["val_acc"]
        axs[0, 1].plot(epochs, val_acc, marker='o', color="blue", label="Validation Accuracy")

        # Find where accuracy stopped improving by more than 0.5% (0.005)
        diffs = np.diff(val_acc)
        under_threshold = np.where(diffs < 0.005)[0]
        plateau_epoch = under_threshold[0] + 2 if under_threshold.size > 0 else None

        if plateau_epoch:
            axs[0, 1].axvline(x=plateau_epoch, color='red', linestyle='--', label=f"Plateau (<0.5%) @ Ep {plateau_epoch}")
            print(f"[Info] Learning may have plateaued around epoch {plateau_epoch}")

        axs[0, 1].set_title("Validation Accuracy Plateau")
        axs[0, 1].set_xlabel("Epochs")
        axs[0, 1].set_ylabel("Accuracy")
        axs[0, 1].legend()
        axs[0, 1].grid(True, linestyle=":")
    else:
        axs[0, 1].text(0.5, 0.5, "Validation accuracy data missing", ha='center', va='center')

    # ----------------------------------------------------
    # PANEL 3: Gradient Norms (Scale Check)
    # ----------------------------------------------------
    if "grad_norms" in history and "grad_norm_std" in history:
        grad_means = history["grad_norms"]
        grad_stds = history["grad_norm_std"]

        axs[1, 0].plot(epochs, grad_means, marker='o', color='teal', label='Mean')
        axs[1, 0].plot(epochs, grad_stds, marker='x', color='orange', linestyle='--', label='Std')
        axs[1, 0].set_title("Gradient Norms (Scale Check)")
        axs[1, 0].set_xlabel("Epochs")
        axs[1, 0].set_ylabel("Norm Value")
        axs[1, 0].legend()
        axs[1, 0].grid(True, linestyle=":")
    else:
        axs[1, 0].text(0.5, 0.5, "Gradient norm metrics missing", ha='center', va='center')

    # ----------------------------------------------------
    # PANEL 4: Coefficient of Variation (Batch Variance)
    # ----------------------------------------------------
    if "grad_norms" in history and "grad_norm_std" in history:
        grad_means = history["grad_norms"]
        grad_stds = history["grad_norm_std"]
        cv_values = [s / m if m > 0 else 0 for m, s in zip(grad_means, grad_stds)]

        axs[1, 1].plot(epochs, cv_values, marker='s', color='purple', label='CV (Std/Mean)')
        axs[1, 1].axhline(y=1.5, color='red', linestyle=':', label='High Variance Trigger (1.5)')
        axs[1, 1].set_title("Coefficient of Variation (CV)")
        axs[1, 1].set_xlabel("Epochs")
        axs[1, 1].set_ylabel("CV Value")
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle=":")
    else:
        axs[1, 1].text(0.5, 0.5, "Gradient variance metrics missing", ha='center', va='center')

    # Tighten layout and render the unified dashboard
    plt.tight_layout()
    plt.show()