import os
import joblib
import torch.nn as nn

import shared.encryption as en


MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
FEDERATEDMODEL_PATH = os.path.join(MODEL_FOLDER, "federated_model.pkl")


def save_federated_model(model):
    """
    Save the federated model to disk.
    """
    joblib.dump(model, FEDERATEDMODEL_PATH)


def load_federated_model():
    """
    Load the federated model from disk, or create a new one if it doesn't exist.
    """
    if os.path.exists(FEDERATEDMODEL_PATH):
        return joblib.load(FEDERATEDMODEL_PATH)
    else:
        model = generate_base_model(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            hidden_layers=HIDDEN_LAYERS,
            activation=ACTIVATION
        )
        save_federated_model(model)
        return model