from pathlib import Path

import polars as pl
from gensim.models import KeyedVectors


def load_slices(aligned_dir: Path) -> dict[str, KeyedVectors]:
    """Load the aligned models in `aligned_dir` (as written by `align_slices`) by label"""
    return {
        path.stem: KeyedVectors.load(str(path))
        for path in sorted(aligned_dir.glob("*.kv"))
    }


def neighbors_over_time(
    slices: dict[str, KeyedVectors], word: str, n: int = 5
) -> pl.DataFrame:
    """The `n` nearest neighbors of `word` in each slice that contains it"""
    rows = [
        (label, rank, neighbor, similarity)
        for label, wv in slices.items()
        if word in wv.key_to_index
        for rank, (neighbor, similarity) in enumerate(
            wv.most_similar(word, topn=n), start=1
        )
    ]
    return pl.DataFrame(
        rows, schema=["label", "rank", "neighbor", "similarity"], orient="row"
    )
