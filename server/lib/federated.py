import os
import torch
from collections import OrderedDict


MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
FEDERATED_MODEL_PATH = os.path.join(MODEL_FOLDER, "federated_model.pkl")
    

def aggregate_models(input_model_paths, output_path):
    """
    Aggregate given models and save new model to given path.
    """
    models = []
    for path in input_model_paths:
        model = torch.load(path, map_location="cpu", weights_only=True)
        models.append(model)

    avg_state_dict = OrderedDict()
    for key in models[0].keys():
        avg_state_dict[key] = torch.zeros_like(models[0][key])

    for state_dict in models:
        for key in state_dict.keys():
            avg_state_dict[key] += state_dict[key]

    num_models = len(models)
    for key in avg_state_dict.keys():
        avg_state_dict[key] /= num_models

    torch.save(model, output_path)