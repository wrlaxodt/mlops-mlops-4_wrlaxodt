from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from src.dataset.data_loader import SimpleDataLoader
from src.dataset.watch_log import get_datasets
from src.model.movie_predictor import MoviePredictor, ModelParams, model_save
from src.evaluate.evaluate import evaluate


@dataclass
class TrainConfig:
    num_epochs: int = 20
    batch_size: int = 64
    model_ext: str = "pkl"


def train_one_epoch(model: MoviePredictor, train_loader: SimpleDataLoader) -> float:
    total_loss = 0.0
    n = 0

    for features, labels in train_loader:
        model.partial_fit(features, labels)
        proba = model.predict_proba(features)  # (B, C)
        # 간단 loss: (1 - 정답클래스확률)의 평균
        correct_p = proba[np.arange(len(labels)), labels]
        loss = float(np.mean(1.0 - correct_p))
        total_loss += loss * len(labels)
        n += len(labels)

    return total_loss / max(n, 1)


def run_train(
    model_name: str = "movie_predictor",
    num_epochs: int = 20,
    batch_size: int = 64,
    model_ext: str = "pkl",
) -> Dict[str, Any]:
    train_dataset, val_dataset, _ = get_datasets()

    model_params = {
        "input_dim": train_dataset.features_dim,
        "num_classes": train_dataset.num_classes,
        "hidden_layer_sizes": (64, 32),
        "random_state": 0,
    }
    model = MoviePredictor(
        input_dim=model_params["input_dim"],
        num_classes=model_params["num_classes"],
        params=ModelParams(hidden_layer_sizes=model_params["hidden_layer_sizes"], random_state=model_params["random_state"]),
    )

    train_loader = SimpleDataLoader(train_dataset.features, train_dataset.labels, batch_size=batch_size, shuffle=True)
    val_loader = SimpleDataLoader(val_dataset.features, val_dataset.labels, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_path = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_pred = evaluate(model, val_loader)

        print(f"[Epoch {epoch}/{num_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = model_save(
                model=model,
                model_params=model_params,
                epoch=epoch,
                loss=val_loss,
                scaler=train_dataset.scaler,
                label_encoder=train_dataset.label_encoder,
                ext=model_ext,
            )

    return {"best_model_path": best_path, "best_val_loss": best_val_loss}
