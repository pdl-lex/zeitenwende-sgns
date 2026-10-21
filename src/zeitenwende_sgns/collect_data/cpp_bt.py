import pandas as pd
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


def load_cpp_bt_corpus() -> pd.DataFrame:
    if not CORPUS_PATH.exists():
        fetch_cpp_bt_corpus()
    return pd.read_parquet(CORPUS_PATH)


df = load_cpp_bt_corpus()

START_YEAR = "2014"
END_YEAR = "2025"

df.index = pd.to_datetime(df.sitzung_datum).rename("date")
df = df.sort_index()

df = df.loc[START_YEAR:END_YEAR, ["rede_text"]].rename(columns={"rede_text": "text"})

df["doc_id"] = df.text.map(create_doc_id)

df = df.drop_duplicates(subset=["doc_id"])


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df[["doc_id", "text"]].to_csv(OUTPUT_DIR / "cpp_bt.csv")
