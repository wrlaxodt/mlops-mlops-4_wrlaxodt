from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from dotenv import load_dotenv
import fire

from src.data_prepare.crawler import TMDBCrawler
from src.data_prepare.preprocessing import TMDBWatchLogGenerator, WatchLogGenConfig


def project_root() -> Path:
    # repo-root/src/data_prepare/main.py -> parents[2] == repo-root
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_data_prepare(
    start_page: int = 1,
    end_page: int = 1,
    region: str = "KR",
    language: str = "ko-KR",
    user_count: int = 100,
    max_select_count: int = 20,
    output_csv: str = "dataset/watch_log.csv",
    save_raw_json: bool = True,
    raw_json_path: str = "dataset/raw/popular.json",
) -> Dict[str, Any]:
    """
    TMDB popular movies -> 가상 유저 watch_log.csv 생성
    """
    load_dotenv()  # 로컬에선 .env 사용, 운영에선 외부 주입 :contentReference[oaicite:11]{index=11}

    crawler = TMDBCrawler.from_env(region=region, language=language)
    movies = crawler.get_bulk_popular_movies(start_page=start_page, end_page=end_page)

    root = project_root()

    if save_raw_json:
        raw_path = root / raw_json_path
        ensure_dir(raw_path.parent)
        raw_path.write_text(json.dumps({"movies": movies}, ensure_ascii=False), encoding="utf-8")

    cfg = WatchLogGenConfig(user_count=user_count, max_select_count=max_select_count)
    gen = TMDBWatchLogGenerator(movies, cfg)
    df = gen.generate()

    out_path = root / output_csv
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)

    return {
        "movies_count": len(movies),
        "watch_log_rows": int(df.shape[0]),
        "output_csv": str(out_path),
        "raw_json": str((root / raw_json_path)) if save_raw_json else None,
    }


def head_watch_log(path: str = "dataset/watch_log.csv", n: int = 5) -> str:
    root = project_root()
    df = pd.read_csv(root / path)
    return df.head(n).to_string(index=False)


if __name__ == "__main__":
    fire.Fire(
        {
            "run": run_data_prepare,
            "head": head_watch_log,
        }
    )

