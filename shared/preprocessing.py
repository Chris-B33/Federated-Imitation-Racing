import pandas as pd
import torch.nn as nn

INPUT_DIM = 8
OUTPUT_DIM = 8
BINARY_OUTPUTS = 7


def generate_base_model(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_layers=[128, 64], activation=nn.ReLU) -> nn.Module:
    """
    Create a base model of set architecture for centralised and federated model to use.
    """
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))

    model = nn.Sequential(*layers)
    return model


def normalise_inputs(df: pd.DataFrame, mean=None, std=None) -> pd.DataFrame:
    """
    Normalise input features using min-max scaling.
    """
    if mean is None or std is None:
        mean = df.mean()
        std = df.std().replace(0, 1)  # avoid division by zero
    
    df_norm = (df - mean) / std
    return df_norm.fillna(0)


def normalise_labels(df: pd.DataFrame, binary_cols_count=7, tilt_mean=None, tilt_std=None):
    """
    Z-score normalize outputs:
      - binary outputs stay 0/1
      - continuous tilt is normalized
    """
    df_norm = df.copy()
    
    df_norm.iloc[:, :binary_cols_count] = df.iloc[:, :binary_cols_count]
    
    tilt_col = df.columns[binary_cols_count]
    
    if tilt_mean is None or tilt_std is None:
        tilt_mean = df[tilt_col].mean()
        tilt_std = df[tilt_col].std() or 1.0
    
    df_norm[tilt_col] = (df[tilt_col] - tilt_mean) / tilt_std
    
    return df_norm.fillna(0)
