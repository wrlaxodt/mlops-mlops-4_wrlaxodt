from __future__ import annotations

from typing import Tuple, List

import numpy as np

from src.dataset.data_loader import SimpleDataLoader
from src.model.movie_predictor import MoviePredictor


def evaluate(model: MoviePredictor, val_loader: SimpleDataLoader) -> Tuple[float, List[int]]:
    total_loss = 0.0
    total_n = 0
    predictions = []

    for features, labels in val_loader:
        proba = model.predict_proba(features)
        pred = np.argmax(proba, axis=1)
        predictions.extend(pred.tolist())

        correct_p = proba[np.arange(len(labels)), labels]
        loss = float(np.mean(1.0 - correct_p))

        total_loss += loss * len(labels)
        total_n += len(labels)

    return total_loss / max(total_n, 1), predictions
