import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gensim.models import KeyedVectors
from matplotlib.figure import Figure


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


def similarity_matrix(slices: dict[str, KeyedVectors], word: str) -> pl.DataFrame:
    """Diachronic similarity matrix of `word` (Kiyama et al. 2025): the cosine
    similarity of its vectors in each pair of slices, with one row and column per
    slice. Entries for slices that do not contain `word` are null."""
    vectors = np.array(
        [
            wv.get_vector(word, norm=True)
            if word in wv.key_to_index
            else np.full(wv.vector_size, np.nan)
            for wv in slices.values()
        ]
    )
    return (
        pl.DataFrame(vectors @ vectors.T, schema=list(slices))
        .fill_nan(None)
        .insert_column(0, pl.Series("label", list(slices)))
    )


def next_label(label: str) -> str:
    """The slice label following `label`, e.g. 2023 for 2022 or 2023-Q1 for 2022-Q4"""
    year, _, part = label.partition("-")
    if not part:
        return str(int(year) + 1)
    kind, number = part[0], int(part[1:])
    if number < {"H": 2, "Q": 4}[kind]:
        return f"{year}-{kind}{number + 1}"
    return f"{int(year) + 1}-{kind}1"


def plot_similarity_matrices(
    slices: dict[str, KeyedVectors], words: list[str], ncols: int = 3
) -> Figure:
    """Draw the `similarity_matrix` of each of `words` as a heatmap on a shared
    color scale. Ticks mark the start of each slice."""
    matrices = [
        similarity_matrix(slices, word).drop("label").to_numpy() for word in words
    ]
    vmin = min(np.nanmin(matrix) for matrix in matrices)
    labels = [*slices, next_label(list(slices)[-1])]
    ticks = np.arange(len(labels)) - 0.5
    tick_labels = [f"'{label[2:]}" for label in labels]

    nrows = math.ceil(len(words) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
        layout="constrained",
    )
    for ax, word, matrix in zip(axes.flat, words, matrices):
        image = ax.imshow(matrix, cmap="RdYlBu_r", vmin=vmin, vmax=1)
        ax.set(
            title=word,
            xticks=ticks,
            yticks=ticks,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
        )
    for ax in axes.flat[len(words) :]:
        ax.set_axis_off()
    fig.colorbar(image, ax=axes, shrink=0.6)
    return fig
