from __future__ import annotations

import datetime
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier

from src.utils.utils import model_dir, save_hash


@dataclass
class ModelParams:
    hidden_layer_sizes: Tuple[int, ...] = (64, 32)
    max_iter: int = 1  # epoch 루프를 직접 돌리려고 1로 두고 partial_fit 사용
    random_state: int = 0


class MoviePredictor:
    """
    input : (rating, popularity, watch_seconds)
    output: content_id(class)
    """
    name = "movie_predictor"

    def __init__(self, input_dim: int, num_classes: int, params: ModelParams = ModelParams()):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.params = params

        self.clf = MLPClassifier(
            hidden_layer_sizes=params.hidden_layer_sizes,
            max_iter=params.max_iter,
            random_state=params.random_state,
            warm_start=False,
        )
        self._classes = np.arange(num_classes)

    def partial_fit(self, x: np.ndarray, y: np.ndarray) -> None:
        # MLPClassifier는 첫 호출에서 classes 필요
        self.clf.partial_fit(x, y, classes=self._classes)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict(x)


def model_save(model: MoviePredictor, model_params: Dict[str, Any], epoch: int, loss: float, scaler, label_encoder, ext: str = "pkl") -> str:
    save_dir = model_dir(model.name)
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    dst = os.path.join(save_dir, f"E{epoch}_T{current_time}.{ext}")

    save_data = {
        "epoch": epoch,
        "model_params": model_params,
        "model": model,  # sklearn 객체 포함
        "loss": loss,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }

    with open(dst, "wb") as f:
        pickle.dump(save_data, f)

    save_hash(dst)
    print(f"Model saved to {dst}")
    return dst
