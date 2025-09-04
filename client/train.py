import requests

def train_model(data_name):
    model_data = b"dummy model weights"
    with open(data_name, "wb") as f:
        f.write(model_data)
    return f"{data_name}.model"

def send_model(path):
    url = "http://server:5000/upload"
    with open(path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    print("Server response:", response.text)

if __name__ == "__main__":
    name = input("Input data file name: ")
    model_file = train_model(data_name=name)
    send_model(path=model_file)