"""
This script compares the most recent federated and centralised model against each other on the same validation data.
It then outputs plots for this data.
"""


import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import shared.preprocessing as pp
import shared.training as tr


MODEL_FOLDER = f"models"
FED_MODEL_PATH = f"{MODEL_FOLDER}/federated_model.pt"
CEN_MODEL_PATH = f"{MODEL_FOLDER}/centralised_model.pt"
VAL_INPUTS_PATH = f"shared/val_data/inputs.csv"
VAL_LABELS_PATH = f"shared/val_data/labels.csv"

BINARY_OUTPUTS = 7


def evaluate_model(model, data_loader):
    model.eval()
    all_metrics = {
        "loss": 0.0,
        "binary_acc": 0.0,
        "steer_mae": 0.0
    }
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_inputs, batch_labels in data_loader:
            preds = model(batch_inputs)
            metrics = tr.compute_metrics(preds, batch_labels, binary_cols_count=BINARY_OUTPUTS)

            all_metrics["loss"] += metrics["loss"]
            all_metrics["binary_acc"] += metrics["binary_acc"]
            all_metrics["steer_mae"] += metrics["steer_mae"]

    # average over batches
    for k in all_metrics:
        all_metrics[k] /= num_batches

    return all_metrics


if __name__ == "__main__":
    # Initialise models
    fed_model = pp.generate_base_model()
    cen_model = pp.generate_base_model()

    # Load both model dicts
    fed_dicts = torch.load(FED_MODEL_PATH, map_location="cpu", weights_only=True)
    cen_dicts = torch.load(CEN_MODEL_PATH, map_location="cpu", weights_only=True)

    # Load model states
    fed_model.load_state_dict(fed_dicts)
    cen_model.load_state_dict(cen_dicts)

    # gather inputs
    inputs = pd.read_csv(VAL_INPUTS_PATH)
    labels = pd.read_csv(VAL_INPUTS_PATH)

    inputs_norm = pp.normalise_inputs(inputs)
    labels_norm = pp.normalise_labels(labels)

    inputs_tensor = torch.tensor(inputs_norm.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_norm.values, dtype=torch.float32)

    # create validation dataset
    dataset = TensorDataset(inputs_tensor, labels_tensor)
    val_loader = DataLoader(dataset, shuffle=False)

    # evaluate models
    fed_res = evaluate_model(fed_model, val_loader)
    cen_res = evaluate_model(cen_model, val_loader)

    # Detailed visualization
    button_names = ["1", "2", "PLUS", "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT"]

    with torch.no_grad():
        # Binary outputs
        fed_binary = (torch.sigmoid(fed_model(inputs_tensor)[:, :BINARY_OUTPUTS]) > 0.5).int()
        cen_binary = (torch.sigmoid(cen_model(inputs_tensor)[:, :BINARY_OUTPUTS]) > 0.5).int()
        true_binary = labels_tensor[:, :BINARY_OUTPUTS].int()

        fed_acc_per_button = (fed_binary == true_binary).float().mean(dim=0).cpu().numpy()
        cen_acc_per_button = (cen_binary == true_binary).float().mean(dim=0).cpu().numpy()

        # STEER outputs
        fed_steer = fed_model(inputs_tensor)[:, BINARY_OUTPUTS].cpu().numpy()
        cen_steer = cen_model(inputs_tensor)[:, BINARY_OUTPUTS].cpu().numpy()

    # Metrics to compare
    metrics_names = ["loss", "binary_acc", "steer_mae"]

    # Values for each model
    fed_values = [fed_res[m] for m in metrics_names]
    cen_values = [cen_res[m] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, fed_values, width, label="Federated", color='skyblue')
    plt.bar(x + width/2, cen_values, width, label="Centralised", color='salmon')

    plt.xticks(x, metrics_names)
    plt.ylabel("Metric Value")
    plt.title("Validation Metrics Comparison")
    plt.legend()
    plt.show()

    # Plot per-button accuracy
    x = np.arange(len(button_names))
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, fed_acc_per_button, width, label="Federated")
    plt.bar(x + width/2, cen_acc_per_button, width, label="Centralised")
    plt.xticks(x, button_names)
    plt.ylabel("Accuracy")
    plt.title("Per-Button Accuracy Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()