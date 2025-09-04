import requests
from datetime import datetime

PATH = "input/data.csv"

def train_model(data_name):
    data = open(data_name, "r+").readlines()

    # train... train... train...

    model_data = b"dummy model weights"

    with open("cur_model.bin", "wb") as f:
        f.write(model_data)
    return "cur_model.bin"

def send_model(path):
    url = "http://server:5000/upload"
    with open(path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    print(f"[+] {datetime.now().strftime('%H:%M:%S')} Server response: {response.text}")

if __name__ == "__main__":
    model_file = train_model(data_name=PATH)
    send_model(path=model_file)