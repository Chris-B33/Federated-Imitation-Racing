from flask import Flask, request, Response, jsonify
import os
import torch
from datetime import datetime

import lib.federated as fe
import lib.utils as ut

import shared.encryption as en
import shared.preprocessing as pp
import shared.training as tr


MODEL_FOLDER = "/app/models"
FED_MODEL_PATH = f"{MODEL_FOLDER}/federated_model.pt"
CENTRAL_MODEL_PATH = f"{MODEL_FOLDER}/centralised_model.pt"
CENTRAL_INPUTS_PATH = f"data/inputs.csv"
CENTRAL_LABELS_PATH = f"data/labels.csv"

app = Flask(__name__)
os.makedirs(MODEL_FOLDER, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    """
    Health check before opening client on compose
    If the global model file doesn't exist, generate it on the fly.
    """
    try:
        if not os.path.exists(FED_MODEL_PATH) or os.path.getsize(FED_MODEL_PATH) == 0:
            model = pp.generate_base_model()
            torch.save(model.state_dict(), FED_MODEL_PATH)
        return jsonify({"status": "ready"}), 200
    except Exception as e:
        return jsonify({"status": "loading", "error": str(e)}), 503


@app.route("/download_model", methods=["GET"])
def download_model():
    """
    Send global model to client and generate basic model if needed.
    """
    try:
        if not os.path.exists(FED_MODEL_PATH) or os.path.getsize(FED_MODEL_PATH) == 0:
            model = pp.generate_base_model()
            torch.save(model.state_dict(), FED_MODEL_PATH)
        
        sd = torch.load(FED_MODEL_PATH, map_location="cpu", weights_only=True)
        
        encoded = en.encode_model(sd)
        return Response(encoded, mimetype="application/octet-stream")

    except Exception as e:
        print(e)
        return str(e), 500


@app.route("/upload_model", methods=["POST"])
def upload_model():
    """
    Receives model from client and saves it to to_be_federated to be used later.
    If there is 3 or more, aggregate them.
    """
    try:
        if "file" not in request.files:
            return "No file uploaded", 400

        encoded = request.files["file"].read()
        sd = en.decode_model(encoded)
        while True:
            name = ut.generate_random_name()
            path = os.path.join(MODEL_FOLDER, f"to_be_federated/{name}.pt")
            if not os.path.exists(path):
                break

        torch.save(sd, F"{MODEL_FOLDER}/to_be_federated/{name}.pt")

        model_files = [f for f in os.listdir(f"{MODEL_FOLDER}/to_be_federated") if f.endswith(".pt")]
        if len(model_files) >= 3:
            print(f"[+][{datetime.now().strftime('%H:%M:%S')}] Enough models to aggregate!", flush=True)
            # Train federated model
            to_aggregate = [os.path.join(f"{MODEL_FOLDER}/to_be_federated", f) for f in model_files]
            aggregated_path = os.path.join(MODEL_FOLDER, "federated_model.pt")
            fe.aggregate_models(to_aggregate, aggregated_path)
            for p in to_aggregate:
                os.remove(p)

            # Train centralised model
            if not os.path.exists(CENTRAL_MODEL_PATH) or os.path.getsize(CENTRAL_MODEL_PATH) == 0:
                central_model = pp.generate_base_model()
                torch.save(central_model.state_dict(), CENTRAL_MODEL_PATH)
        
            central_model = torch.load(CENTRAL_MODEL_PATH, map_location="cpu", weights_only=True)
            central_model = tr.update_model(
                central_model,
                CENTRAL_INPUTS_PATH,
                CENTRAL_LABELS_PATH
            )
            torch.save(central_model.state_dict(), CENTRAL_MODEL_PATH)

            # Remove server dataset
            os.remove(CENTRAL_INPUTS_PATH)
            os.remove(CENTRAL_LABELS_PATH)

            return f"Model: {name} was uploaded, enough models found to aggregate", 200
        else:
            return f"Model: {name} is uploaded", 200
        
    except Exception as e:
        print(e)
        return str(e), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
