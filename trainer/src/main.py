from __future__ import annotations

import fire
import numpy as np
from dotenv import load_dotenv

from src.data_prepare.main import run_data_prepare
from src.train.train import run_train
from src.inference.inference import (
    load_checkpoint,
    init_model,
    inference as run_inference,
)


def preprocessing(date: str = "25-05-12"):
    load_dotenv()
    print(f"Run date : {date}")
    return run_data_prepare(
        start_page=1,
        end_page=1,
        output_csv="dataset/watch_log.csv",
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
