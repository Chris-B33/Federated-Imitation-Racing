from flask import Flask, request, Response, jsonify
from datetime import datetime
import io
import os
import logging
import torch
import torch.nn as nn

import lib.federated as fe
import shared.encryption as en
import shared.preprocessing as pp


MODEL_FOLDER = "/app/models"
GLOBAL_MODEL_PATH = f"{MODEL_FOLDER}/federated_model.pt"

app = Flask(__name__)
os.makedirs(MODEL_FOLDER, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    """
    Health check before opening client on compose
    If the global model file doesn't exist, generate it on the fly.
    """
    try:
        if not os.path.exists(GLOBAL_MODEL_PATH) or os.path.getsize(GLOBAL_MODEL_PATH) == 0:
            model = pp.generate_base_model()
            torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
        return jsonify({"status": "ready"}), 200
    except Exception as e:
        return jsonify({"status": "loading", "error": str(e)}), 503


@app.route("/download_model", methods=["GET"])
def download_model():
    """
    Send global model to client and generate basic model if needed.
    """
    try:
        if not os.path.exists(GLOBAL_MODEL_PATH) or os.path.getsize(GLOBAL_MODEL_PATH) == 0:
            model = pp.generate_base_model()
            torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
        
        sd = torch.load(GLOBAL_MODEL_PATH, map_location="cpu", weights_only=True)
        
        encoded = en.encode_model(sd)
        return Response(encoded, mimetype="application/octet-stream")

    except Exception as e:
        print(e)
        return str(e), 500


@app.route("/upload_model", methods=["POST"])
def upload_model():
    try:
        if "file" not in request.files:
            return "No file uploaded", 400

        encoded = request.files["file"].read()
        sd = en.decode_model(encoded)
        torch.save(sd, GLOBAL_MODEL_PATH)

        return "Model uploaded", 200
    except Exception as e:
        print(e)
        return str(e), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
