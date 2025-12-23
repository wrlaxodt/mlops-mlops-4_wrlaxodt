from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class TMDBConfig:
    base_url: str
    api_key: str
    region: str = "KR"
    language: str = "ko-KR"
    request_interval_seconds: float = 0.4
    timeout_seconds: int = 30
    max_retries: int = 3

  
    """
    TMDB 'movie-popular-list' API 기반 크롤러
    popular 호출, region=KR, language=ko-KR, page 파라미터 사용 :contentReference[oaicite:1]{index=1} 
    세션 재사용 + 재시도 + rate-limit sleep + 예외 메시지 강화
    """
  
class TMDBCrawler:

    def __init__(self, config: TMDBConfig):
        if not config.base_url:
            raise ValueError("TMDB_BASE_URL is empty (check .env / env vars).")
        if not config.api_key:
            raise ValueError("TMDB_API_KEY is empty (check .env / env vars).")

        self.config = config
        self.session = requests.Session()

    @classmethod
    def from_env(
        cls,
        region: str = "KR",
        language: str = "ko-KR",
        request_interval_seconds: float = 0.4,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ) -> "TMDBCrawler":
        base_url = os.environ.get("TMDB_BASE_URL", "https://api.themoviedb.org/3/movie").strip()
        api_key = os.environ.get("TMDB_API_KEY", "").strip()

        cfg = TMDBConfig(
            base_url=base_url,
            api_key=api_key,
            region=region,
            language=language,
            request_interval_seconds=request_interval_seconds,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        return cls(cfg)

    def _request(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        base_params = {
            "api_key": self.config.api_key,
            "language": self.config.language,
            "region": self.config.region,
        }
        merged = {**base_params, **params}

        last_err: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.get(url, params=merged, timeout=self.config.timeout_seconds)
                if resp.status_code == 200:
                    return resp.json()

                # 429/5xx는 재시도 가치가 큼
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self.config.request_interval_seconds * attempt)
                    last_err = RuntimeError(f"TMDB HTTP {resp.status_code}: {resp.text[:200]}")
                    continue

                # 그 외는 즉시 실패로 처리
                raise RuntimeError(f"TMDB HTTP {resp.status_code}: {resp.text[:400]}")

            except Exception as e:
                last_err = e
                time.sleep(self.config.request_interval_seconds * attempt)

        raise RuntimeError(f"TMDB request failed after retries. last_error={last_err}")

    def get_popular_movies(self, page: int) -> List[Dict[str, Any]]:
        payload = self._request("/popular", {"page": page})
        return payload.get("results", []) or []

    def get_bulk_popular_movies(self, start_page: int, end_page: int) -> List[Dict[str, Any]]:
        movies: List[Dict[str, Any]] = []
        for page in range(start_page, end_page + 1):
            movies.extend(self.get_popular_movies(page))
            time.sleep(self.config.request_interval_seconds)
        return movies

