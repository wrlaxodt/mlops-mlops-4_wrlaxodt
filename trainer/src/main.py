from __future__ import annotations

import fire
import numpy as np
from dotenv import load_dotenv

from src.data_prepare.preprocessing import run_data_prepare

from src.train.train import run_train
from src.inference.inference import (
    load_checkpoint,
    init_model,
    inference as run_inference,
)


def preprocessing(
    output_csv: str = "dataset/watch_log.csv",
    user_count: int = 100,
    max_select_count: int = 20,
    seed: int = 0,
):
    """
    IMDb TSV 기반 watch_log.csv 생성
    """
    load_dotenv()
    return run_data_prepare(
        output_csv=output_csv,
        user_count=user_count,
        max_select_count=max_select_count,
        seed=seed,
    )


def train(
    model_name: str = "movie_predictor",
    num_epochs: int = 20,
    batch_size: int = 64,
    model_ext: str = "pkl",
):

    return run_train(
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        model_ext=model_ext,
    )


def inference(
    data: str = "",
    batch_size: int = 1,
    model_name: str = "movie_predictor",
    model_ext: str = "pkl",
):
    """
    단건 / 배치 추론
    """
    load_dotenv()

    checkpoint = load_checkpoint(model_name=model_name, model_ext=model_ext)
    model, scaler, label_encoder = init_model(checkpoint)

    if data.strip():
        x = np.array(eval(data))
    else:
        x = np.array([])

    result = run_inference(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        data=x,
        batch_size=batch_size,
    )
    print(result)
    return result


if __name__ == "__main__":
    fire.Fire(
        {
            "preprocessing": preprocessing,
            "train": train,
            "inference": inference,
        }
    )
