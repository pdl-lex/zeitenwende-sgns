from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
CORPUS_DIR = DATA_DIR / "corpus"
MODELS_DIR = DATA_DIR / "models"
ALIGNED_DIR = DATA_DIR / "aligned"

RESOURCES_DIR = PROJECT_ROOT / "resources"

TARGETS_PATH = RESOURCES_DIR / "zielwoerter.txt"
LEMMA_OVERRIDES_PATH = RESOURCES_DIR / "lemma_overrides.tsv"
