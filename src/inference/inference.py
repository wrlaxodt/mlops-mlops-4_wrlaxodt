from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from typing import List

from src.utils.utils import (
    get_latest_model,
    calculate_hash,
    read_hash,
)
from src.dataset.watch_log import WatchLogDataset, get_datasets
from src.dataset.data_loader import SimpleDataLoader
from src.evaluate.evaluate import evaluate


def model_validation(model_path: str) -> bool:
    """
    모델 무결성 검증 (sha256)
    """
    original_hash = read_hash(model_path)
    current_hash = calculate_hash(model_path)

    if original_hash == current_hash:
        print("model hash validation success")
        return True
    else:
        print("model hash validation failed")
        return False


def load_checkpoint(model_name: str = "movie_predictor", model_ext: str = "pkl") -> dict:
    """
    가장 최신 모델 로드 + hash 검증
    """
    latest_model = get_latest_model(model_name, ext=model_ext)

    if not model_validation(latest_model):
        raise FileExistsError("Invalid model file (hash mismatch)")

    with open(latest_model, "rb") as f:
        checkpoint = pickle.load(f)

    return checkpoint


def init_model(checkpoint: dict):
    """
    checkpoint에서 모델 + scaler + label_encoder 복원
    """
    model = checkpoint["model"]
    scaler = checkpoint["scaler"]
    label_encoder = checkpoint["label_encoder"]

    return model, scaler, label_encoder


def make_inference_df(data: np.ndarray) -> pd.DataFrame:
    """
    단건 추론용 DataFrame 생성
    실습 기준 컬럼 순서 유지
    """
    columns = ["user_id", "content_id", "watch_seconds", "rating", "popularity"]
    return pd.DataFrame(data=[data], columns=columns)


def inference(
    model,
    scaler,
    label_encoder,
    data: np.ndarray,
    batch_size: int = 1,
) -> List[str]:
    """
    - data가 있으면: 단건 추론
    - data가 비어있으면: 전체 test dataset 기준 batch inference
    """
    if data.size > 0:
        df = make_inference_df(data)
        dataset = WatchLogDataset(df, scaler=scaler, label_encoder=label_encoder)
    else:
        _, _, dataset = get_datasets(scaler=scaler, label_encoder=label_encoder)

    dataloader = SimpleDataLoader(
        dataset.features,
        dataset.labels,
        batch_size=batch_size,
        shuffle=False,
    )

    loss, predictions = evaluate(model, dataloader)
    print(f"inference loss: {loss}")

    # content_id decode
    results = [dataset.decode_content_id(idx) for idx in predictions]
    return results

