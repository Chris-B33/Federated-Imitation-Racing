import os
import torch
from collections import OrderedDict


MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
FEDERATED_MODEL_PATH = os.path.join(MODEL_FOLDER, "federated_model.pkl")
    

def aggregate_models(input_model_paths, output_path, epsilon=0.01):
    """
    Aggregate given models and save new model to given path.
    """
    models = []
    for path in input_model_paths:
        model = torch.load(path, map_location="cpu", weights_only=True)
        models.append(model)

    # Create zero dict
    avg_state_dict = OrderedDict()
    for key in models[0].keys():
        avg_state_dict[key] = torch.zeros_like(models[0][key])

    # Compute norms
    norms = []
    for model in models:
        norm = torch.sqrt(sum(torch.sum(param ** 2) for param in model.values()))
        norms.append(norm + epsilon)  # avoid zero

    # Aggregate models weighted by their norms
    agg_state_dict = OrderedDict()
    for key in models[0].keys():
        weighted_sum = sum(norm * model[key] for norm, model in zip(norms, models))
        agg_state_dict[key] = weighted_sum / sum(norms)

    torch.save(model, output_path)