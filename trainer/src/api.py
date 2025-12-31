# src/api.py
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.inference import (
    load_checkpoint,
    init_model,
    inference,
)

app = FastAPI()

# 서버 시작 시 모델 1번만 로드
checkpoint = load_checkpoint(model_name="movie_predictor")
model, scaler, label_encoder = init_model(checkpoint)


class PredictRequest(BaseModel):
    user_id: int
    content_id: int
    watch_seconds: int
    rating: float
    popularity: float


@app.post("/predict")
def predict(req: PredictRequest):
    data = np.array([
        req.user_id,
        req.content_id,
        req.watch_seconds,
        req.rating,
        req.popularity,
    ])

    result = inference(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        data=data,
    )

    return {
        "recommended_content_id": result
    }

