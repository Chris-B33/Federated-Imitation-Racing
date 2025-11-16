import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import shared.preprocessing as pp


OPTIMISER = torch.optim.Adam
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 256


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, binary_cols_count: int):
    """
    Compute metrics for mixed outputs: binary buttons + continuous steer.
    """
    preds_binary = preds[:, :binary_cols_count]
    preds_steer = preds[:, binary_cols_count:]
    labels_binary = labels[:, :binary_cols_count]
    labels_steer = labels[:, binary_cols_count:]

    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_cont = nn.MSELoss()
    loss_binary = criterion_binary(preds_binary, labels_binary)
    loss_steer = criterion_cont(preds_steer.squeeze(), labels_steer.squeeze())
    total_loss = loss_binary + loss_steer

    with torch.no_grad():
        preds_bin_sig = torch.sigmoid(preds_binary)
        pred_labels = (preds_bin_sig > 0.5).float()
        binary_acc = (pred_labels == labels_binary).float().mean().item()

        steer_mae = torch.mean(torch.abs(preds_steer.squeeze() - labels_steer.squeeze())).item()

    return {
        "loss": total_loss,
        "binary_acc": binary_acc,
        "steer_mae": steer_mae
    }


def update_model(model, inputs_path, labels_path, optimiser=OPTIMISER, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    """
    Train the model on mini-batches with gradient clipping.
    Metrics are computed using compute_metrics().
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    inputs = pd.read_csv(inputs_path)
    labels = pd.read_csv(labels_path)

    inputs_norm = pp.normalise_inputs(inputs)
    labels_norm = pp.normalise_labels(labels)

    inputs_tensor = torch.tensor(inputs_norm.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_norm.values, dtype=torch.float32)

    dataset = TensorDataset(inputs_tensor, labels_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(train_loader)

    instantiated_optimiser = optimiser(model.parameters(), lr=learning_rate)
    binary_cols_count = labels_tensor.shape[1] - 1

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_binary_acc = 0
        epoch_steer_mae = 0
        
        for i, (batch_inputs, batch_labels) in enumerate(train_loader):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            preds = model(batch_inputs)
            metrics = compute_metrics(preds, batch_labels, binary_cols_count)

            instantiated_optimiser.zero_grad()
            total_loss = metrics["loss"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            instantiated_optimiser.step()

            epoch_loss += metrics["loss"]
            epoch_binary_acc += metrics["binary_acc"]
            epoch_steer_mae += metrics["steer_mae"]

            print(
                f"Epoch {(epoch+1):02d} | "
                f"Batch: {(i+1):02d}/{num_batches} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Binary Acc: {metrics['binary_acc']:.4f} | "
                f"Steer MAE: {metrics['steer_mae']:.4f}",
            )

        print(
            f"Epoch {epoch+1}/{epochs} | " +
            f"Loss: {epoch_loss/num_batches:.4f} | " +
            f"Binary Acc: {epoch_binary_acc/num_batches:.4f} | " +
            f"Steer MAE: {epoch_steer_mae/num_batches:.4f}",
            end="\n"
        )

    return model
