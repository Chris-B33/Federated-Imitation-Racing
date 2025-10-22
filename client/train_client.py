import requests
from datetime import datetime

from lib import preprocessing as pp

INPUTS_PATH = "input/inputs.csv"
LABELS_PATH = "input/labels.csv"

def get_global_model():
    url = "http://server:5000/download_model"
    response = requests.get(server_url)
    return response.files

def train_model(model, inputs_path, labels_path):
    inputs = open(inputs_path, "r+").readlines()
    labels = open(labels_path, "r+").readlines()

    normalised_inputs = pp.normalise_inputs(inputs)
    normalised_labels = pp.normalise_labels(labels)

    # train... train... train...
    
    updated_model = b"dawdadw"
    return updated_model

def send_file(file):
    url = "http://server:5000/upload_model"
    files = {"file": file}
    response = requests.post(url, files=files)

def main():
    print(f"[+][{datetime.now().strftime('%H:%M:%S')}] Getting global model...")
    model = get_global_model()

    print(f"[+][{datetime.now().strftime('%H:%M:%S')}] Training model...")
    model = train_model(
        model=model,
        inputs_path=INPUTS_PATH,
        labels_path=LABELS_PATH
    )

    print(f"[+][{datetime.now().strftime('%H:%M:%S')}] Sending model to server...")
    send_file(model=model)


if __name__ == "__main__":
    main()
    