import os
import requests
from datetime import datetime

import shared.preprocessing as pp
import shared.encryption as en
import shared.training as tr


SERVER_URL  = "http://server:5000"

INPUTS_PATH = "data/inputs.csv"
LABELS_PATH = "data/labels.csv"


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


def send_model(model):
    """
    Send trained model to server
    """
    encoded = en.encode_model(model.state_dict())
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
        model = tr.update_model(
            model=model,
            inputs_path=INPUTS_PATH,
            labels_path=LABELS_PATH
        )

        print(f"[+][{datetime.now().strftime('%H:%M:%S')}] Sending model to server...", flush=True)
        send_model(model)

        # del data after
        os.remove("data/inputs.csv")
        os.remove("data/labels.csv")

    except Exception as e:
        print(f"[+][{datetime.now().strftime('%H:%M:%S')}][ERROR] {e}", flush=True)


if __name__ == "__main__":
    main()
