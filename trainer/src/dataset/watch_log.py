from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder



def project_path() -> str:
    """
    repo-root 경로를 반환
    (utils가 없거나 경로가 꼬였을 때도 동작하도록 최소 구현)
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
    )


class WatchLogDataset:
    """
    input  : 평점(rating), 인기도(popularity), 시청 시간(watch_seconds)
    output : 추천 콘텐츠(content_id)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: Optional[StandardScaler] = None,
        label_encoder: Optional[LabelEncoder] = None,
    ):
        self.df = df.copy()
        self.scaler = scaler
        self.label_encoder = label_encoder

        self.features: np.ndarray = np.empty((0, 0))
        self.labels: np.ndarray = np.empty((0,))
        self.contents_id_map: Dict[int, str] = {}

        self._preprocessing()

    def _preprocessing(self) -> None:
        # content_id를 정수형으로 변환 (LabelEncoder) :contentReference[oaicite:8]{index=8}
        if self.label_encoder is not None:
            self.df["content_id"] = self.label_encoder.transform(self.df["content_id"].astype(str))
        else:
            self.label_encoder = LabelEncoder()
            self.df["content_id"] = self.label_encoder.fit_transform(self.df["content_id"].astype(str))

        # 디코딩 맵 생성 :contentReference[oaicite:9]{index=9}
        self.contents_id_map = dict(enumerate(self.label_encoder.classes_))

        # 타겟(label) & 피처(feature) 정의 :contentReference[oaicite:10]{index=10}
        feature_cols = ["rating", "popularity", "watch_seconds"]
        self.labels = self.df["content_id"].values.astype(np.int64)
        raw_features = self.df[feature_cols].values.astype(np.float32)

        # 스케일링 :contentReference[oaicite:11]{index=11}
        if self.scaler is not None:
            self.features = self.scaler.transform(raw_features)
        else:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(raw_features)

    def decode_content_id(self, encoded_id: int) -> str:
        return self.contents_id_map[int(encoded_id)]

    @property
    def features_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.label_encoder.classes_)  # :contentReference[oaicite:12]{index=12}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def read_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.join(project_path(), "dataset", "watch_log.csv")
    return pd.read_csv(csv_path)


def split_dataset(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=seed)
    return train_df, val_df, test_df


def get_datasets(
    scaler: Optional[StandardScaler] = None,
    label_encoder: Optional[LabelEncoder] = None,
    csv_path: Optional[str] = None,
) -> Tuple[WatchLogDataset, WatchLogDataset, WatchLogDataset]:
    """
    train_dataset이 fit한 scaler/label_encoder를 val/test에 재사용 :contentReference[oaicite:15]{index=15}
    """
    df = read_dataset(csv_path=csv_path)
    train_df, val_df, test_df = split_dataset(df)

    train_dataset = WatchLogDataset(train_df, scaler=scaler, label_encoder=label_encoder)
    val_dataset = WatchLogDataset(val_df, scaler=train_dataset.scaler, label_encoder=train_dataset.label_encoder)
    test_dataset = WatchLogDataset(test_df, scaler=train_dataset.scaler, label_encoder=train_dataset.label_encoder)

    return train_dataset, val_dataset, test_dataset
