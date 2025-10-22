from flask import Flask, request
from datetime import datetime
import os

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/download_model", methods=["GET"])
def download_global_model():
    pass

@app.route("/upload_model", methods=["POST"])
def upload_model():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]

@app.route("/aggregate_models", methods=["POST"])
def aggregate_models():
    pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)