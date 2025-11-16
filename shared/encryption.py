import io
import base64
import torch

def encode_model(state_dict: dict) -> bytes:
    """
    Encode a PyTorch state_dict into base64 bytes.
    """
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read())

def decode_model(encoded_bytes: bytes) -> dict:
    """
    Decode base64 bytes back into a PyTorch state_dict.
    """
    raw = base64.b64decode(encoded_bytes)
    buffer = io.BytesIO(raw)
    return torch.load(buffer, map_location="cpu", weights_only=False)
