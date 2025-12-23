from __future__ import annotations

import os
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.utils.utils import get_latest_model, calculate_hash, read_hash
from src.dataset.watch_log import WatchLogDataset, get_datasets
from src.dataset.data_loader import SimpleDataLoader
from src.evaluate.evaluate import evaluate


def model_validation(model_path: str) -> bool:
    original_hash = read_hash(model_path)
    current_hash = calculate_hash(model_path)
    if original_hash == current_hash:
        print("validation success")
        return True
    return False


def load_checkpoint(model_name: str = "movie_predictor", model_ext: str = "pkl") -> dict:
    latest_model = get_latest_model(model_name, ext=model_ext)
    if not model_validation(latest_model):
        raise FileExistsError("Not found or invalid model file")
    with open(latest_model, "rb") as f:
        checkpoint = pickle.load(f)
    return checkpoint


def init_model(checkpoint: dict):
    model = checkpoint["model"]
    scaler = checkpoint["scaler"]
    label_encoder = checkpoint["label_encoder"]
    return model, scaler, label_encoder


def make_inference_df(data: np.ndarray) -> pd.DataFrame:
    columns = "user_id content_id watch_seconds rating popularity".split()
    return pd.DataFrame(data=[data], columns=columns)


def inference(model, scaler, label_encoder, data: np.ndarray, batch_size: int = 1) -> List[str]:
    if data.size > 0:
        df = make_inference_df(data)
        dataset = WatchLogDataset(df, scaler=scaler, label_encoder=label_encoder)
    else:
        _, _, dataset = get_datasets(scaler=scaler, label_encoder=label_encoder)

    dataloader = SimpleDataLoader(dataset.features, dataset.labels, batch_size=batch_size, shuffle=False)
    loss, predictions = evaluate(model, dataloader)
    print(loss, predictions)

    return [dataset.decode_content_id(idx) for idx in predictions]
