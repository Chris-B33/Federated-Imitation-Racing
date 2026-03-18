"""
Model script acts as server.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # allows for import of "shared"

import socket
import torch
import pandas as pd

import shared.preprocessing as pp

MODEL_NAME = "centralised_model"
MODEL_PATH = f"models/{MODEL_NAME}.pt"

HOST = "127.0.0.1"
PORT = 5000


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print(f"Listening on {HOST}:{PORT}")
conn, addr = server.accept()
print("Connected:", addr)

model = pp.generate_base_model()
dicts = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(dicts)


def main():
    train_min = None
    train_max = None

    while True:
        data = conn.recv(1024)
        if not data:
            break
        
        parts = data.decode().split()
        telemetry = [float(x) for x in parts[:-1]] + [int(parts[-1])]

        telemetry_tensor = torch.tensor(telemetry, dtype=torch.float32).unsqueeze(0)

        if train_min is None:
            train_min = telemetry_tensor.clone()
            train_max = telemetry_tensor.clone()
        else:
            train_min = torch.minimum(train_min, telemetry_tensor)
            train_max = torch.maximum(train_max, telemetry_tensor)

        normalised_telemetry = (telemetry_tensor - train_min) / (train_max - train_min + 1e-8)

        with torch.no_grad():
            outputs = model(normalised_telemetry)

        reply = " ".join(list(map(str, outputs.tolist()[0])))
        conn.sendall(reply.encode())

    conn.close()

if __name__ == "__main__":
    main()