import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_default_config = {
    "data_file_path": "data/wastewater_level.csv",
    "date_col": "Datetime",
    "target_col": "wastewater_level",
    "forecast_horizon": 1,
    "test_split_date": "2023-01-01",
    "train_val_frac": 0.8,
}

rnn_default_config = {
    "hidden_size": 20,
    "num_rnn_layers": 1,
}

transformer_default_config = {
    "d_model": 12,
    "num_enc_layers": 1,
    "nhead": 1,
}

search_space = {
    "common": {
        "quant_bits": {"low": 8, "high": 8, "step": 2},
        "batch_size": {"low": 16, "high": 256, "step": 16},
        "lr": {"low": 1e-5, "high": 1e-3, "log": True},
    },
    "lstm": {
        "hidden_size": {"low": 8, "high": 64, "step": 8},
    },
    "transformer": {
        "d_model": {"low": 8, "high": 64, "step": 8},
    },
}
