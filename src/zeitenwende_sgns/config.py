from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
MODELS_DIR = DATA_DIR / "models"

TARGETS_PATH = DATA_DIR / "masterliste_zielwörter_okt26.csv"
LEMMA_OVERRIDES_PATH = DATA_DIR / "lemma_overrides.csv"
