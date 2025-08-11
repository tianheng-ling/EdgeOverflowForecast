import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, mode: str, data_config: dict):
        self.data_path = data_config["data_file_path"]
        self.date_col = data_config["date_col"]
        self.target_col = data_config["target_col"]
        self.test_split_date = data_config["test_split_date"]
        self.train_val_frac = data_config["train_val_frac"]
        self.window_size = int(data_config["window_size"])
        self.forecast_horizon = data_config["forecast_horizon"]
        self.mode = mode  # ['train', 'val' or 'test']

        self.train_scaled, self.val_scaled, self.test_scaled, self.scaler = (
            self._load_and_preprocess_data()
        )

        if self.mode == "train":
            self.data = self.train_scaled
        elif self.mode == "val":
            self.data = self.val_scaled
        elif self.mode == "test":
            self.data = self.test_scaled
        else:
            raise ValueError("Mode must be 'train', 'val' or 'test'")

    def _load_and_preprocess_data(self):
        df = pd.read_csv(
            self.data_path, parse_dates=[self.date_col], index_col=self.date_col
        )

        train_val_df = df.loc[: self.test_split_date, self.target_col]
        test_df = df.loc[self.test_split_date :, self.target_col]

        train_val_data = train_val_df.values.reshape(-1, 1)
        test_data = test_df.values.reshape(-1, 1)

        train_size = int(len(train_val_data) * self.train_val_frac)
        train_data = train_val_data[:train_size]
        val_data = train_val_data[train_size:]

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        val_scaled = scaler.transform(val_data)
        test_scaled = scaler.transform(test_data)

        return train_scaled, val_scaled, test_scaled, scaler

    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_horizon + 1

    def __getitem__(self, idx):
        X = self.data[idx : idx + self.window_size]
        y = self.data[
            idx + self.window_size : idx + self.window_size + self.forecast_horizon
        ]

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
