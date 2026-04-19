"""
Compares federated and centralised imitation learning models across all rounds
on validation data. Produces console metrics and figures for FYP reporting.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import shared.preprocessing as pp
import shared.training as tr


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_FOLDER    = "models/rounds"
NUM_ROUNDS      = 5
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


def evaluate_model(model, inputs_t, labels_t):
    model.eval()
    loader = DataLoader(TensorDataset(inputs_t, labels_t), shuffle=False)
    totals = {"loss": 0.0, "binary_acc": 0.0, "steer_mae": 0.0}
    with torch.no_grad():
        for x, y in loader:
            m = tr.compute_metrics(model(x), y, binary_cols_count=BINARY_OUTPUTS)
            for k in totals:
                totals[k] += m[k]
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def get_predictions(model, inputs_t, norm_stats):
    model.eval()
    with torch.no_grad():
        out = model(inputs_t)
        binary = (torch.sigmoid(out[:, :BINARY_OUTPUTS]) > 0.5).int().cpu().numpy()
        steer  = out[:, BINARY_OUTPUTS].cpu().numpy()
        if norm_stats:
            steer = steer * norm_stats["tilt_std"] + norm_stats["tilt_mean"]
    return binary, steer


def print_summary(round_idx, fed_res, cen_res, fed_mae_raw, cen_mae_raw):
    print(f"\n  Round {round_idx}")
    print("=" * 57)
    print(f"  {'Metric':<30} {'Federated':>10} {'Centralised':>11}")
    print("=" * 57)
    print(f"  {'Loss':<30} {fed_res['loss']:>10.4f} {cen_res['loss']:>11.4f}")
    print(f"  {'Button Accuracy':<30} {fed_res['binary_acc']:>9.1%} {cen_res['binary_acc']:>10.1%}")
    print(f"  {'Steer MAE (raw units)':<30} {fed_mae_raw:>10.3f} {cen_mae_raw:>11.3f}")
    print("=" * 57)


# ── Load val data ─────────────────────────────────────────────────────────────

inputs_raw  = pd.read_csv(VAL_INPUTS_PATH)
labels_raw  = pd.read_csv(VAL_LABELS_PATH)
true_steer  = labels_raw[labels_raw.columns[BINARY_OUTPUTS]].values
true_binary = labels_raw.iloc[:, :BINARY_OUTPUTS].values.astype(int)

# ── Evaluate all rounds ───────────────────────────────────────────────────────

rounds = list(range(1, NUM_ROUNDS + 1))

fed_history = {"loss": [], "binary_acc": [], "steer_mae_raw": []}
cen_history = {"loss": [], "binary_acc": [], "steer_mae_raw": []}

for r in rounds:
    fed_model, fed_stats = load_model(f"{MODEL_FOLDER}/federated_model_round{r}.pt")
    cen_model, cen_stats = load_model(f"{MODEL_FOLDER}/centralised_model_round{r}.pt")

    fed_inputs_t, fed_labels_t = make_tensors(inputs_raw, labels_raw, fed_stats)
    cen_inputs_t, cen_labels_t = make_tensors(inputs_raw, labels_raw, cen_stats)

    fed_res = evaluate_model(fed_model, fed_inputs_t, fed_labels_t)
    cen_res = evaluate_model(cen_model, cen_inputs_t, cen_labels_t)

    _, fed_steer_r = get_predictions(fed_model, fed_inputs_t, fed_stats)
    _, cen_steer_r = get_predictions(cen_model, cen_inputs_t, cen_stats)

    fed_mae_raw = float(np.mean(np.abs(fed_steer_r - true_steer)))
    cen_mae_raw = float(np.mean(np.abs(cen_steer_r - true_steer)))

    print_summary(r, fed_res, cen_res, fed_mae_raw, cen_mae_raw)

    for hist, res, mae in [(fed_history, fed_res, fed_mae_raw),
                           (cen_history, cen_res, cen_mae_raw)]:
        hist["loss"].append(res["loss"])
        hist["binary_acc"].append(res["binary_acc"])
        hist["steer_mae_raw"].append(mae)

# Reuse final-round model objects for per-frame figures
fed_binary, fed_steer = get_predictions(fed_model, fed_inputs_t, fed_stats)
cen_binary, cen_steer = get_predictions(cen_model, cen_inputs_t, cen_stats)
frames = np.arange(len(true_steer))


# ── Figure 1: Overall metrics (final round) ───────────────────────────────────
fed_final_res = {"loss": fed_history["loss"][-1], "binary_acc": fed_history["binary_acc"][-1]}
cen_final_res = {"loss": cen_history["loss"][-1], "binary_acc": cen_history["binary_acc"][-1]}
fed_final_mae = fed_history["steer_mae_raw"][-1]
cen_final_mae = cen_history["steer_mae_raw"][-1]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(f"Overall Validation Metrics (Round {NUM_ROUNDS})", fontweight="bold")

for ax, (title, fv, cv, fmt) in zip(axes, [
    ("Loss",            fed_final_res["loss"],       cen_final_res["loss"],       ".4f"),
    ("Button Accuracy", fed_final_res["binary_acc"], cen_final_res["binary_acc"], ".1%"),
    ("Steer MAE (raw)", fed_final_mae,               cen_final_mae,               ".3f"),
]):
    bars = ax.bar(["Federated", "Centralised"], [fv, cv],
                  color=[FED_COLOR, CEN_COLOR], width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt=f"%{fmt}", padding=4)
    ax.set_title(title)
    ax.set_ylim(0, max(fv, cv) * 1.3 or 0.01)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()


# ── Figure 2: Metrics over rounds ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Validation Metrics Across Training Rounds", fontweight="bold")

for ax, (title, key, pct) in zip(axes, [
    ("Loss",            "loss",          False),
    ("Button Accuracy", "binary_acc",    True),
    ("Steer MAE (raw)", "steer_mae_raw", False),
]):
    ax.plot(rounds, fed_history[key], marker="o", color=FED_COLOR, label="Federated",   linewidth=1.8)
    ax.plot(rounds, cen_history[key], marker="s", color=CEN_COLOR, label="Centralised", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_xticks(rounds)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    if pct:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.tight_layout()
plt.show()


# ── Figure 3 (old Fig 4): Steering over time + smoothed error (final round) ───
fed_err = np.abs(fed_steer - true_steer)
cen_err = np.abs(cen_steer - true_steer)
window  = max(1, len(frames) // 60)
smooth  = lambda x: np.convolve(x, np.ones(window) / window, mode="same")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle(f"Steering: Prediction vs Ground Truth (Round {NUM_ROUNDS})", fontweight="bold")

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


# ── Figure 4 (old Fig 5): Steering distribution (final round) ────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle(f"Steering Prediction Distribution (Round {NUM_ROUNDS})", fontweight="bold")
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
