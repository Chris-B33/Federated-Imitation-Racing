import requests
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import shared.preprocessing as pp
import shared.encryption as en


INPUTS_PATH = "data/inputs.csv"
LABELS_PATH = "data/labels.csv"
SERVER_URL  = "http://server:5000"

OPTIMISER = torch.optim.Adam
LEARNING_RATE = 1e-5
CRITERION = nn.MSELoss
EPOCHS = 25
BATCH_SIZE = 256


def get_global_model():
    """
    Get global model from server
    """
    url = f"{SERVER_URL}/download_model"
    response = requests.get(url)
    response.raise_for_status()
    
    sd_bytes = response.content
    sd = en.decode_model(sd_bytes)

    model = pp.generate_base_model()
    model.load_state_dict(sd)

    return model


def update_model(model, inputs_path, labels_path, optimiser, criterion, epochs, batch_size):
    """device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    updated_model = model

    inputs = pd.read_csv(inputs_path).values
    labels = pd.read_csv(labels_path).values

    normalised_inputs = pp.normalise_inputs(inputs)
    normalised_labels = pp.normalise_labels(labels)

    inputs_tensor = torch.tensor(normalised_inputs, dtype=torch.float32)
    labels_tensor = torch.tensor(normalised_labels, dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            loss = criterion()(preds, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return updated_model"""
    return model


def send_model(model):
    """
    Send trained model to server
    """
    encoded = en.encode_model(model)
    url = f"{SERVER_URL}/upload_model"
    files = {"file": ("model.pt", encoded)}
    response = requests.post(url, files=files)
    response.raise_for_status()
    print(f"[+][{datetime.now().strftime('%H:%M:%S')}] Model sent successfully!", flush=True)


def main():
    """
    Main function of Client
    """
    try:
        print(f"[+][{datetime.now().strftime('%H:%M:%S')}] Getting global model...", flush=True)
        model = get_global_model()

        print(f"[+][{datetime.now().strftime('%H:%M:%S')}] Training model...", flush=True)
        model = update_model(
            model=model,
            inputs_path=INPUTS_PATH,
            labels_path=LABELS_PATH,
            optimiser=OPTIMISER(model.parameters(), lr=LEARNING_RATE),
            criterion=CRITERION,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        print(f"[+][{datetime.now().strftime('%H:%M:%S')}] Sending model to server...", flush=True)
        send_model(model)

    except Exception as e:
        print(f"[+][{datetime.now().strftime('%H:%M:%S')}][ERROR] {e}", flush=True)


if __name__ == "__main__":
    main()
