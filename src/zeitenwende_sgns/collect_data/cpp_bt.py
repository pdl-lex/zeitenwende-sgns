from datetime import date

import polars as pl
import requests
from tqdm import tqdm

from zeitenwende_sgns.config import PDF_DIR, RAW_DIR
from zeitenwende_sgns.util import create_doc_id

CORPUS_DIR = PDF_DIR / "cpp_bt"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_FILE = "CPP-BT_2026-01-17_DE_PQT_Reden_Gesamt.parquet"
CORPUS_PATH = CORPUS_DIR / CORPUS_FILE

OUTPUT_DIR = RAW_DIR / "cpp_bt"


def fetch_cpp_bt_corpus():
    """Download the Bundestag speeches corpus"""
    corpus_url = f"https://zenodo.org/records/18177196/files/{CORPUS_FILE}?download=1"

    response = requests.get(corpus_url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(CORPUS_PATH, "wb") as f,
        tqdm(
            desc=f"Downloading {CORPUS_FILE}",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            size = f.write(chunk)
            bar.update(size)


def load_cpp_bt_corpus() -> pl.LazyFrame:
    if not CORPUS_PATH.exists():
        fetch_cpp_bt_corpus()
    return pl.scan_parquet(CORPUS_PATH)


if __name__ == "__main__":
    START_YEAR = 2014
    END_YEAR = 2025

    df = (
        load_cpp_bt_corpus()
        .select(pl.col("sitzung_datum").alias("date"), pl.col("rede_text").alias("text"))
        .filter(pl.col("date").is_between(date(START_YEAR, 1, 1), date(END_YEAR, 12, 31)))
        .sort("date", maintain_order=True)
        .with_columns(
            pl.col("text").map_elements(create_doc_id, return_dtype=pl.String).alias("doc_id")
        )
        .unique(subset="doc_id", keep="first", maintain_order=True)
        .select("date", "doc_id", "text")
        .collect()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.write_parquet(OUTPUT_DIR / "cpp_bt.parquet")
