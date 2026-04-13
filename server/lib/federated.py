import os
import torch
from collections import OrderedDict


MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
FEDERATED_MODEL_PATH = os.path.join(MODEL_FOLDER, "federated_model.pkl")
    

def aggregate_models(input_model_paths, output_path, epsilon=0.01):
    """
    Aggregate given models and save new model to given path.
    Norm stats are averaged across clients and embedded in the output bundle.
    """
    models = []
    all_norm_stats = []
    for path in input_model_paths:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" in loaded:
            models.append(loaded["state_dict"])
            all_norm_stats.append(loaded["norm_stats"])
        else:
            models.append(loaded)

    # Compute norms
    norms = []
    for model in models:
        norm = torch.sqrt(sum(torch.sum(param ** 2) for param in model.values()))
        norms.append(norm + epsilon)  # avoid zero

    # Aggregate model weights weighted by their norms
    agg_state_dict = OrderedDict()
    for key in models[0].keys():
        weighted_sum = sum(norm * model[key] for norm, model in zip(norms, models))
        agg_state_dict[key] = weighted_sum / sum(norms)

    # Average norm stats across clients if available
    if all_norm_stats:
        n = len(all_norm_stats)
        avg_norm_stats = {
            "input_mean": [sum(s["input_mean"][i] for s in all_norm_stats) / n for i in range(len(all_norm_stats[0]["input_mean"]))],
            "input_std":  [sum(s["input_std"][i]  for s in all_norm_stats) / n for i in range(len(all_norm_stats[0]["input_std"]))],
            "tilt_mean":  sum(s["tilt_mean"] for s in all_norm_stats) / n,
            "tilt_std":   sum(s["tilt_std"]  for s in all_norm_stats) / n,
        }
        torch.save({"state_dict": agg_state_dict, "norm_stats": avg_norm_stats}, output_path)
    else:
        torch.save(agg_state_dict, output_path)