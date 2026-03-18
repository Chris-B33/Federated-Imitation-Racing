import pandas as pd
import torch.nn as nn


def generate_base_model(input_dim=7, output_dim=8, hidden_layers=[128, 64], activation=nn.ReLU) -> nn.Module:
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


def normalise_inputs(df: pd.DataFrame, col_min=None, col_max=None) -> pd.DataFrame:
    """
    Normalise input features using min-max scaling.

    If col_min and col_max are provided (from training data),
    they will be used instead of computing them from df.
    """

    if col_min is None or col_max is None:
        col_min = df.min()
        col_max = df.max()

    df_norm = (df - col_min) / (col_max - col_min)

    return df_norm.fillna(0)


def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise output labels.
    For multi-output regression: scale non-binary columns.
    """
    df_norm = df.copy()
    
    for col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max != col_min:
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.0
    return df_norm.fillna(0)
