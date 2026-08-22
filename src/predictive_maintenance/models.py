from __future__ import annotations

from typing import Any


def build_lstm(input_size: int) -> Any:
    import torch

    class RULModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size, 64, num_layers=2, dropout=0.2, batch_first=True)
            self.head = torch.nn.Linear(64, 1)

        def forward(self, x: Any) -> Any:
            output, _ = self.lstm(x)
            return self.head(output[:, -1]).squeeze(-1)

    return RULModel()


def xgboost_grid() -> list[dict[str, float | int]]:
    return [
        {
            "max_depth": depth,
            "learning_rate": rate,
            "n_estimators": trees,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        for depth in (3, 6)
        for rate in (0.03, 0.1)
        for trees in (300, 600)
    ]
