from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
MODELS_DIR = DATA_DIR / "models"

RESOURCES_DIR = PROJECT_ROOT / "resources"

TARGETS_PATH = RESOURCES_DIR / "zielwoerter.tsv"
LEMMA_OVERRIDES_PATH = RESOURCES_DIR / "lemma_overrides.tsv"
