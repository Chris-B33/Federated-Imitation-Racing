"""
Compares the federated and centralised imitation learning models on validation data.
Produces console metrics and five figures suitable for FYP reporting and demo.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import shared.preprocessing as pp
import shared.training as tr


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_FOLDER    = "models"
FED_MODEL_PATH  = f"{MODEL_FOLDER}/federated_model.pt"
CEN_MODEL_PATH  = f"{MODEL_FOLDER}/centralised_model.pt"
VAL_INPUTS_PATH = "shared/val_data/inputs.csv"
VAL_LABELS_PATH = "shared/val_data/labels.csv"

BINARY_OUTPUTS = pp.BINARY_OUTPUTS
BUTTON_NAMES   = ["2", "1", "PLUS", "D-Up", "D-Down", "D-Left", "D-Right"]

FED_COLOR = "#4C9BE8"
CEN_COLOR = "#E8644C"
GT_COLOR  = "#4CAF50"

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(path):
    """Load a model and its embedded norm stats from a plain state dict or bundle."""
    model = pp.generate_base_model()
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in bundle:
        model.load_state_dict(bundle["state_dict"])
        norm_stats = bundle["norm_stats"]
    else:
        model.load_state_dict(bundle)
        norm_stats = None
        print(f"Warning: {path} has no embedded norm stats")
    return model, norm_stats


def make_tensors(inputs, labels, norm_stats):
    """Normalise validation data using the model's own training stats."""
    if norm_stats is not None:
        input_mean = pd.Series(norm_stats["input_mean"], index=inputs.columns)
        input_std  = pd.Series(norm_stats["input_std"],  index=inputs.columns)
        inputs_norm = pp.normalise_inputs(inputs, mean=input_mean, std=input_std)
        labels_norm = pp.normalise_labels(labels, tilt_mean=norm_stats["tilt_mean"], tilt_std=norm_stats["tilt_std"])
    else:
        inputs_norm = pp.normalise_inputs(inputs)
        labels_norm = pp.normalise_labels(labels)
    return (
        torch.tensor(inputs_norm.values, dtype=torch.float32),
        torch.tensor(labels_norm.values, dtype=torch.float32),
    )


def evaluate_model(model, data_loader):
    model.eval()
    totals = {"loss": 0.0, "binary_acc": 0.0, "steer_mae": 0.0}
    n = len(data_loader)
    with torch.no_grad():
        for inputs, labels in data_loader:
            m = tr.compute_metrics(model(inputs), labels, binary_cols_count=BINARY_OUTPUTS)
            for k in totals:
                totals[k] += m[k]
    return {k: v / n for k, v in totals.items()}


def print_summary(fed_res, cen_res, fed_steer_mae_raw, cen_steer_mae_raw):
    print("\n" + "=" * 57)
    print(f"  {'Metric':<30} {'Federated':>10} {'Centralised':>11}")
    print("=" * 57)
    print(f"  {'Loss':<30} {fed_res['loss']:>10.4f} {cen_res['loss']:>11.4f}")
    print(f"  {'Button Accuracy':<30} {fed_res['binary_acc']:>9.1%} {cen_res['binary_acc']:>10.1%}")
    print(f"  {'Steer MAE (raw units)':<30} {fed_steer_mae_raw:>10.3f} {cen_steer_mae_raw:>11.3f}")
    print("=" * 57 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load models
    fed_model, fed_stats = load_model(FED_MODEL_PATH)
    cen_model, cen_stats = load_model(CEN_MODEL_PATH)

    # Load raw validation data
    inputs = pd.read_csv(VAL_INPUTS_PATH)
    labels = pd.read_csv(VAL_LABELS_PATH)

    # Normalise per-model using each model's own training stats
    fed_inputs_t, fed_labels_t = make_tensors(inputs, labels, fed_stats)
    cen_inputs_t, cen_labels_t = make_tensors(inputs, labels, cen_stats)

    # Evaluate
    fed_res = evaluate_model(fed_model, DataLoader(TensorDataset(fed_inputs_t, fed_labels_t), shuffle=False))
    cen_res = evaluate_model(cen_model, DataLoader(TensorDataset(cen_inputs_t, cen_labels_t), shuffle=False))

    # Run full forward passes
    with torch.no_grad():
        fed_out = fed_model(fed_inputs_t)
        cen_out = cen_model(cen_inputs_t)

        # Binary button predictions (0/1)
        fed_binary = (torch.sigmoid(fed_out[:, :BINARY_OUTPUTS]) > 0.5).int().cpu().numpy()
        cen_binary = (torch.sigmoid(cen_out[:, :BINARY_OUTPUTS]) > 0.5).int().cpu().numpy()
        # Binary ground truth is unaffected by label normalisation (stays 0/1)
        true_binary = fed_labels_t[:, :BINARY_OUTPUTS].int().cpu().numpy()

        # Steering — denormalised back to raw units
        fed_steer  = fed_out[:, BINARY_OUTPUTS].cpu().numpy()
        cen_steer  = cen_out[:, BINARY_OUTPUTS].cpu().numpy()
        if fed_stats:
            fed_steer = fed_steer * fed_stats["tilt_std"] + fed_stats["tilt_mean"]
        if cen_stats:
            cen_steer = cen_steer * cen_stats["tilt_std"] + cen_stats["tilt_mean"]

    true_steer = labels[labels.columns[BINARY_OUTPUTS]].values

    # Raw steer MAE
    fed_steer_mae_raw = float(np.mean(np.abs(fed_steer - true_steer)))
    cen_steer_mae_raw = float(np.mean(np.abs(cen_steer - true_steer)))

    print_summary(fed_res, cen_res, fed_steer_mae_raw, cen_steer_mae_raw)

    frames = np.arange(len(true_steer))


    # ── Figure 1: Overall metrics ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Overall Validation Metrics", fontweight="bold")

    metric_configs = [
        ("Loss",              fed_res["loss"],        cen_res["loss"],        ".4f"),
        ("Button Accuracy",   fed_res["binary_acc"],  cen_res["binary_acc"],  ".1%"),
        ("Steer MAE (raw)",   fed_steer_mae_raw,      cen_steer_mae_raw,      ".3f"),
    ]

    for ax, (title, fv, cv, fmt) in zip(axes, metric_configs):
        bars = ax.bar(["Federated", "Centralised"], [fv, cv],
                      color=[FED_COLOR, CEN_COLOR], width=0.5, edgecolor="white")
        ax.bar_label(bars, fmt=f"%{fmt}", padding=4)
        ax.set_title(title)
        ax.set_ylim(0, max(fv, cv) * 1.3 or 0.01)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()


    # ── Figure 2: Per-button accuracy ────────────────────────────────────────
    fed_acc_per = (fed_binary == true_binary).mean(axis=0)
    cen_acc_per = (cen_binary == true_binary).mean(axis=0)

    x = np.arange(len(BUTTON_NAMES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Per-Button Prediction Accuracy", fontweight="bold")
    b1 = ax.bar(x - width / 2, fed_acc_per, width, label="Federated",   color=FED_COLOR)
    b2 = ax.bar(x + width / 2, cen_acc_per, width, label="Centralised", color=CEN_COLOR)
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(BUTTON_NAMES)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.18)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


    # ── Figure 3: Button press rates vs ground truth ─────────────────────────
    true_rates = true_binary.mean(axis=0)
    fed_rates  = fed_binary.mean(axis=0)
    cen_rates  = cen_binary.mean(axis=0)

    x = np.arange(len(BUTTON_NAMES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Button Press Rate vs Ground Truth", fontweight="bold")
    ax.bar(x - width, true_rates, width, label="Ground Truth", color=GT_COLOR,  alpha=0.9)
    ax.bar(x,         fed_rates,  width, label="Federated",    color=FED_COLOR, alpha=0.9)
    ax.bar(x + width, cen_rates,  width, label="Centralised",  color=CEN_COLOR, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(BUTTON_NAMES)
    ax.set_ylabel("Press Rate")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


    # ── Figure 4: Steering over time + smoothed error ────────────────────────
    fed_err = np.abs(fed_steer - true_steer)
    cen_err = np.abs(cen_steer - true_steer)
    window  = max(1, len(frames) // 60)
    smooth  = lambda x: np.convolve(x, np.ones(window) / window, mode="same")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Steering: Prediction vs Ground Truth", fontweight="bold")

    ax1.plot(frames, true_steer, label="Ground Truth", color=GT_COLOR,  alpha=0.85, linewidth=1.3)
    ax1.plot(frames, fed_steer,  label="Federated",    color=FED_COLOR, alpha=0.75, linewidth=1.0)
    ax1.plot(frames, cen_steer,  label="Centralised",  color=CEN_COLOR, alpha=0.75, linewidth=1.0)
    ax1.set_ylabel("Steering Value")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(frames, smooth(fed_err), label="Federated error",   color=FED_COLOR, linewidth=1.3)
    ax2.plot(frames, smooth(cen_err), label="Centralised error", color=CEN_COLOR, linewidth=1.3)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Abs Error (smoothed)")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()


    # ── Figure 5: Steering distribution ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Steering Prediction Distribution", fontweight="bold")
    bins = 35
    ax.hist(true_steer, bins=bins, alpha=0.55, label="Ground Truth", color=GT_COLOR,  density=True)
    ax.hist(fed_steer,  bins=bins, alpha=0.55, label="Federated",    color=FED_COLOR, density=True)
    ax.hist(cen_steer,  bins=bins, alpha=0.55, label="Centralised",  color=CEN_COLOR, density=True)
    ax.set_xlabel("Steering Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
