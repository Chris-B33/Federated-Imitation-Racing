import io
import base64
import joblib

import torch
import torch.nn as nn


def encode_model(model) -> bytes:
    """
    Serialize a Python model object into base64-encoded bytes.
    Suitable for sending via HTTP or saving compactly.
    """
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read())
    return encoded


def decode_model(encoded_model: bytes) -> nn.Module:
    """
    Decode base64-encoded bytes back into a Python model object.
    """
    raw = base64.b64decode(encoded_model)
    buffer = io.BytesIO(raw)
    model = joblib.load(buffer)
    return model