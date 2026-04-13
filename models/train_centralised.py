import os
import torch
import shared.preprocessing as pp
import shared.training as tr

MODEL_PATH   = "models/centralised_model.pt"
INPUTS_PATH  = "server/data/inputs.csv"
LABELS_PATH  = "server/data/labels.csv"

model = pp.generate_base_model()

if os.path.exists(MODEL_PATH):
    bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    weights = bundle["state_dict"] if "state_dict" in bundle else bundle
    model.load_state_dict(weights)
    print("Loaded existing model.")
else:
    print("No existing model found, training from scratch.")

model, norm_stats = tr.update_model(model, INPUTS_PATH, LABELS_PATH)

torch.save({"state_dict": model.state_dict(), "norm_stats": norm_stats}, MODEL_PATH)
print(f"Saved to {MODEL_PATH}")
