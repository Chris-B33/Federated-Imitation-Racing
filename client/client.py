import requests
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn

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

    decoded_state_dict = en.decode_model(response.content)

    model = pp.generate_base_model()
    model.load_state_dict(decoded_state_dict)
    return model


def update_model(model, inputs_path, labels_path, optimiser, criterion, epochs, batch_size):
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
