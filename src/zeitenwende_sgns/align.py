from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

from zeitenwende_sgns.config import ALIGNED_DIR, MODELS_DIR
from zeitenwende_sgns.split import replace_dir


def normalize(wv: KeyedVectors) -> KeyedVectors:
    """Scale all vectors of `wv` to unit length in place"""
    wv.vectors = wv.get_normed_vectors()
    wv.fill_norms(force=True)
    return wv


def procrustes(base: KeyedVectors, other: KeyedVectors) -> KeyedVectors:
    """Rotate the unit-length vectors of `other` in place onto `base` with orthogonal
    Procrustes, fitted on the words both share (Hamilton et al. 2016). Words that
    only occur in `other` are rotated as well."""
    common = [word for word in other.index_to_key if word in base.key_to_index]
    if not common:
        raise ValueError("No common vocabulary to align on")
    u, _, vt = np.linalg.svd(other[common].T @ base[common])
    other.vectors = other.vectors @ (u @ vt)
    other.fill_norms(force=True)
    return other


def align_slices(models_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Align the models in `models_dir` (as written by `train_slices`) in order of
    their labels, each onto the already aligned previous one, and save them with
    unit-length vectors as `<label>.kv` to `output_dir`. An existing `output_dir`
    is replaced. Returns the aligned models by label."""
    paths = sorted(models_dir.glob("*.kv"))
    base = None
    with replace_dir(output_dir) as tmp_dir:
        for path in tqdm(paths, unit="slice"):
            wv = normalize(KeyedVectors.load(str(path)))
            if base is not None:
                procrustes(base, wv)
            wv.save(str(tmp_dir / path.name), separately=[])
            base = wv

    return {path.stem: output_dir / path.name for path in paths}


if __name__ == "__main__":
    SOURCE = "cpp_bt"
    PERIOD = "quarter"

    align_slices(MODELS_DIR / SOURCE / PERIOD, ALIGNED_DIR / SOURCE / PERIOD)
