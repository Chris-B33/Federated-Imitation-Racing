import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import shared.preprocessing as pp


OPTIMISER = torch.optim.Adam
LEARNING_RATE = 1e-4
EPOCHS = 25
BATCH_SIZE = 256


import torch
import torch.nn as nn

def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, binary_cols_count: int, loss_weights=(1.0, 0.5)):
    """
    Compute losses and metrics for mixed outputs:
      - binary buttons
      - continuous tilt
    """
    preds_binary = preds[:, :binary_cols_count]
    preds_steer = preds[:, binary_cols_count:]
    
    labels_binary = labels[:, :binary_cols_count]
    labels_steer = labels[:, binary_cols_count:]
    
    # Losses
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_cont = nn.MSELoss()
    
    loss_binary = criterion_binary(preds_binary, labels_binary)
    loss_steer = criterion_cont(preds_steer, labels_steer)
    
    # Weighted sum
    total_loss = loss_weights[0] * loss_binary + loss_weights[1] * loss_steer
    
    with torch.no_grad():
        # Binary accuracy
        preds_bin_sig = torch.sigmoid(preds_binary)
        pred_labels = (preds_bin_sig > 0.5).float()
        binary_acc = (pred_labels == labels_binary).float().mean().item()
        
        # Steer MAE
        steer_mae = torch.mean(torch.abs(preds_steer - labels_steer)).item()
    
    return {
        "loss": total_loss,
        "binary_acc": binary_acc,
        "steer_mae": steer_mae
    }


def scale_model_weights(model, framecount, lap_completion, target_frame=3500, target_lap=3, scale_min=0.8, scale_max=1.2):
    """
    Scale all model weights based on framecount/lap_completion performance.
    Keeps relative weight ratios intact.
    """
    # Compute target ratio and current ratio
    target_ratio = target_frame / target_lap
    current_ratio = framecount / lap_completion
    
    # Compute scale factor and clip
    scale_factor = max(scale_min, min(scale_max, current_ratio / target_ratio))
    
    # Apply scaling to all parameters
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(scale_factor)
    
    print(f"Scaling model by factor: {scale_factor:.4f} based on performance metrics")
    return model


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

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_binary_acc = 0
        epoch_steer_mae = 0
        
        for i, (batch_inputs, batch_labels) in enumerate(train_loader):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            preds = model(batch_inputs)
            labels_norm = pp.normalise_labels(labels, binary_cols_count=7)
            metrics = compute_metrics(preds, batch_labels, binary_cols_count=7, loss_weights=(1.0, 0.5))

            instantiated_optimiser.zero_grad()
            total_loss = metrics["loss"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            instantiated_optimiser.step()

            epoch_loss += metrics["loss"].item()
            epoch_binary_acc += metrics["binary_acc"]
            epoch_steer_mae += metrics["steer_mae"]

            print(
                f"Epoch {(epoch+1):02d}/{epochs} | "
                f"Batch: {(i+1):02d}/{num_batches} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Binary Acc: {metrics['binary_acc']:.4f} | "
                f"Steer MAE: {metrics['steer_mae']:.4f}",
            )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {epoch_loss/num_batches:.4f} | "
            f"Binary Acc: {epoch_binary_acc/num_batches:.4f} | "
            f"Steer MAE: {epoch_steer_mae/num_batches:.4f}",
        )

    """model = scale_model_weights(
        model,
        framecount=inputs.iloc[-1]['framecount'],
        lap_completion=inputs.iloc[-1]['lap_completion'],
        target_frame=3500,
        target_lap=3,
        scale_min=0.5,
        scale_max=1.5
    )"""

    return model
