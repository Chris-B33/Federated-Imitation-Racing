from flask import Flask, request
from datetime import datetime
import os

from lib.utils import random_name_generator

UPLOAD_FOLDER = "models"

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]

    random_name = random_name_generator()
    filepath = os.path.join(UPLOAD_FOLDER, random_name)

    file.save(filepath)
    print(f"[+] {datetime.now().strftime('%H:%M:%S')} All Models: {os.listdir(UPLOAD_FOLDER)}")

    return f"Model saved as {random_name}", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)