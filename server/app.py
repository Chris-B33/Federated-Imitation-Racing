from flask import Flask, request
import os

UPLOAD_FOLDER = "models"

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return f"Model saved to {filepath}", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)