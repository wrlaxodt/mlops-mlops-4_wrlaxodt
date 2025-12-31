from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WatchLogGenConfig:
    user_count: int = 100
    max_select_count: int = 20
    max_runtime_seconds: int = 7200  # 2 hours
    seed: int = 0
    # rating이 높을수록 더 많이 선택되게 하는 증강 강도 :contentReference[oaicite:3]{index=3}
    rating_weight_base: float = 2.0


class TMDBWatchLogGenerator:
   
    def __init__(self, movies: List[Dict[str, Any]], config: WatchLogGenConfig = WatchLogGenConfig()):
        self.movies = movies
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)

    def _movie_features(self) -> pd.DataFrame:
        rows = []
        for m in self.movies:
            rows.append(
                {
                    "content_id": str(m.get("id")),
                    "rating": float(m.get("vote_average", 0.0)),
                    "popularity": float(m.get("popularity", 0.0)),
                }
            )
        df = pd.DataFrame(rows).dropna()
        if df.empty:
            raise ValueError("No movies to process (TMDB results empty).")
        return df

    def _weights_from_rating(self, ratings: np.ndarray) -> np.ndarray:
        clipped = np.clip(ratings, 0.0, 10.0)
        w = np.power(self.config.rating_weight_base, clipped)
        # 모두 0이면 균등분포로
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

            # 가중치 기반 샘플링 (replacement=True)
            idx = np.random.choice(len(feat), size=select_count, replace=True, p=probs)
            selected = feat.iloc[idx].reset_index(drop=True)

            for _, r in selected.iterrows():
                rating = float(r["rating"])
                out_rows.append(
                    {
                        "user_id": str(user_id),
                        "content_id": str(r["content_id"]),
                        "watch_seconds": int(self._watch_seconds_from_rating(rating)),
                        "rating": rating,
                        "popularity": float(r["popularity"]),
                    }
                )

        df = pd.DataFrame(out_rows)
        return df

