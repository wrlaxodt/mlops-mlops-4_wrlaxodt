from __future__ import annotations

import os
import gzip
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WatchLogGenConfig:
    user_count: int = 100
    max_select_count: int = 20
    max_runtime_seconds: int = 7200  # 2 hours
    seed: int = 0
    # rating이 높을수록 더 많이 선택되게 하는 증강 강도
    rating_weight_base: float = 2.0
    # 너무 희귀한 작품(투표수 적음) 제외
    min_votes: int = 50
    # 학습/실습용으로 너무 크면 샘플링
    sample_n_titles: int = 20000


class IMDBWatchLogGenerator:
    """
    IMDb public datasets(tsv.gz) 기반 Watch Log 생성기

    - content_id: tconst
    - rating: averageRating
    - popularity: numVotes
    - watch_seconds: rating 기반으로 시뮬레이션
    """

    def __init__(self, ratings_path: str, basics_path: Optional[str] = None, config: WatchLogGenConfig = WatchLogGenConfig()):
        self.ratings_path = ratings_path
        self.basics_path = basics_path
        self.config = config

        random.seed(config.seed)
        np.random.seed(config.seed)

    def _read_tsv_gz(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with gzip.open(path, "rt", encoding="utf-8") as f:
            df = pd.read_csv(f, sep="\t", low_memory=False)
        return df

    def _movie_features(self) -> pd.DataFrame:
        # title.ratings.tsv.gz
        ratings = self._read_tsv_gz(self.ratings_path)

        # 필수 컬럼 확인
        required = {"tconst", "averageRating", "numVotes"}
        missing = required - set(ratings.columns)
        if missing:
            raise ValueError(f"IMDb ratings file missing columns: {missing}")

        df = ratings.rename(
            columns={"tconst": "content_id", "averageRating": "rating", "numVotes": "popularity"}
        )[["content_id", "rating", "popularity"]].copy()

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
        df = df.dropna()

        # 너무 투표수 적은 항목 제거(노이즈 완화)
        df = df[df["popularity"] >= self.config.min_votes].copy()

        # title.basics.tsv.gz로 영화(movie)만 필터
        if self.basics_path and os.path.exists(self.basics_path):
            basics = self._read_tsv_gz(self.basics_path)
            if "tconst" in basics.columns and "titleType" in basics.columns:
                basics = basics.rename(columns={"tconst": "content_id"})
                movie_ids = basics[basics["titleType"] == "movie"][["content_id"]].drop_duplicates()
                df = df.merge(movie_ids, on="content_id", how="inner")

        if df.empty:
            raise ValueError("No titles to process (IMDb results empty after filtering).")

        # 샘플링
        if self.config.sample_n_titles and len(df) > self.config.sample_n_titles:
            df = df.sample(n=self.config.sample_n_titles, random_state=self.config.seed).reset_index(drop=True)

        return df.reset_index(drop=True)

    def _weights_from_rating(self, ratings: np.ndarray) -> np.ndarray:
        clipped = np.clip(ratings, 0.0, 10.0)
        w = np.power(self.config.rating_weight_base, clipped)

        if np.sum(w) <= 0:
            w = np.ones_like(w)

        return w / np.sum(w)

    def _watch_seconds_from_rating(self, rating: float) -> int:
        base = 1.1
        noise_level = 0.1

        max_sec = self.config.max_runtime_seconds
        base_time = max_sec * ((base ** (rating - 5) - base ** -5) / (base ** 5 - base ** -5))
        noise = np.random.normal(0, noise_level * base_time)
        watch_second = int(np.clip(base_time + noise, 0, max_sec))
        return watch_second

    def generate(self) -> pd.DataFrame:
        feat = self._movie_features()
        probs = self._weights_from_rating(feat["rating"].values)

        out_rows: List[Dict[str, Any]] = []
        users = range(1, self.config.user_count + 1)

        for user_id in users:
            select_count = random.randint(0, self.config.max_select_count)
            if select_count == 0:
                continue

            idx = np.random.choice(len(feat), size=select_count, replace=True, p=probs)
            selected = feat.iloc[idx].reset_index(drop=True)

            for _, r in selected.iterrows():
                rating = float(r["rating"])
                out_rows.append(
                    {
                        "user_id": str(user_id),
                        "content_id": str(r["content_id"]),     # IMDb tconst
                        "watch_seconds": int(self._watch_seconds_from_rating(rating)),
                        "rating": rating,                        # IMDb averageRating
                        "popularity": float(r["popularity"]),    # IMDb numVotes
                    }
                )

        return pd.DataFrame(out_rows)


def run_data_prepare(
    output_csv: str = "dataset/watch_log.csv",
    user_count: int = 100,
    max_select_count: int = 20,
    seed: int = 0,
) -> str:
    ratings_path = os.environ.get("IMDB_RATINGS_PATH", "dataset/raw/imdb/title.ratings.tsv.gz")
    basics_path = os.environ.get("IMDB_BASICS_PATH", "dataset/raw/imdb/title.basics.tsv.gz")

    cfg = WatchLogGenConfig(
        user_count=user_count,
        max_select_count=max_select_count,
        seed=seed,
    )
    gen = IMDBWatchLogGenerator(ratings_path=ratings_path, basics_path=basics_path, config=cfg)

    df = gen.generate()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


