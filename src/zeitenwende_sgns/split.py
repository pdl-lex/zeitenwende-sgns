import shutil
from collections.abc import Collection, Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Literal

import polars as pl
from tqdm import tqdm

from zeitenwende_sgns.config import CORPUS_DIR, PREPROCESSED_DIR

Period = Literal["year", "half", "quarter"]

DEFAULT_EXCLUDE_UPOS = frozenset({"PUNCT", "SYM", "SPACE"})

# Add to `exclude_upos` for a corpus without stopwords
FUNCTION_WORD_UPOS = frozenset({"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"})

YEAR = r"^(1[89]|20)\d\d$"
NUMBER = r"^\d+([.,]\d+)*\.?$"


@contextmanager
def replace_dir(output_dir: Path) -> Generator[Path]:
    """Yield a temporary directory that replaces `output_dir` once the block
    finishes without error, so a crashed run leaves no half-written output"""
    tmp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    yield tmp_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    tmp_dir.rename(output_dir)


def period_label(period: Period) -> pl.Expr:
    """Time slice label from the `date` column, e.g. 2022, 2022-H1 or 2022-Q1"""
    date = pl.col("date")
    year = date.dt.year().cast(pl.String)
    if period == "year":
        return year
    if period == "half":
        return pl.concat_str(year, pl.lit("-H"), (date.dt.month() - 1) // 6 + 1)
    if period == "quarter":
        return pl.concat_str(year, pl.lit("-Q"), date.dt.quarter())
    raise ValueError(f"Unknown period: {period}")


def mask_numbers(lemma: pl.Expr) -> pl.Expr:
    """Replace lemmas that look like years (1800-2099) with <year> and other
    numbers (e.g. 3, 3,5, 1.000 or the ordinal 1.) with <num>"""
    return (
        pl.when(lemma.str.contains(YEAR))
        .then(pl.lit("<year>"))
        .when(lemma.str.contains(NUMBER))
        .then(pl.lit("<num>"))
        .otherwise(lemma)
    )


def shard_sentences(
    path: Path,
    period: Period,
    exclude_upos: Collection[str],
    min_tokens: int,
    keep_numbers: bool,
) -> pl.DataFrame:
    """One row per sentence with its time slice label and its lemmas joined by
    spaces, leaving out tokens with an excluded UPOS tag. Unless `keep_numbers` is
    set, numbers are replaced by placeholders (see `mask_numbers`)."""
    token = pl.element()
    lemma = token.filter(
        ~token.struct.field("upos").is_in(list(exclude_upos))
    ).struct.field("lemma")
    return (
        pl.read_parquet(path, columns=["date", "text_preprocessed"])
        .drop_nulls("text_preprocessed")
        .select(
            period_label(period).alias("label"),
            pl.col("text_preprocessed").alias("sentence"),
        )
        .explode("sentence")
        .with_columns(
            pl.col("sentence").list.eval(lemma if keep_numbers else mask_numbers(lemma))
        )
        .filter(pl.col("sentence").list.len() >= min_tokens)
        .with_columns(pl.col("sentence").list.join(" "))
    )


def split(
    input_dir: Path,
    output_dir: Path,
    period: Period = "year",
    exclude_upos: Collection[str] = DEFAULT_EXCLUDE_UPOS,
    min_tokens: int = 2,
    keep_numbers: bool = False,
) -> dict[str, Path]:
    """Split the preprocessed shards in `input_dir` into time slices and write one
    gensim `corpus_file` per slice to `output_dir` (one sentence of lemmas per
    line). Numbers are masked unless `keep_numbers` is set. An existing `output_dir`
    is replaced. Returns the files by label."""
    with replace_dir(output_dir) as tmp_dir, ExitStack() as stack:
        files = {}
        for path in tqdm(sorted(input_dir.glob("*.parquet")), unit="shard"):
            sentences = shard_sentences(
                path, period, exclude_upos, min_tokens, keep_numbers
            )
            for (label,), group in sentences.partition_by(
                "label", as_dict=True, maintain_order=True
            ).items():
                if label not in files:
                    files[label] = stack.enter_context(
                        open(tmp_dir / f"{label}.txt", "w", encoding="utf-8")
                    )
                files[label].write("\n".join(group["sentence"]) + "\n")

    return {label: output_dir / f"{label}.txt" for label in sorted(files)}


def combine(corpus_dirs: Collection[Path], output_dir: Path) -> dict[str, Path]:
    """Concatenate the corpus files of several sources (e.g. the `split` output of
    each source for the same period) per label, for a model over all sources"""
    labels = sorted({path.stem for d in corpus_dirs for path in d.glob("*.txt")})
    with replace_dir(output_dir) as tmp_dir:
        for label in labels:
            with open(tmp_dir / f"{label}.txt", "wb") as out:
                for corpus_dir in corpus_dirs:
                    path = corpus_dir / f"{label}.txt"
                    if path.exists():
                        with open(path, "rb") as f:
                            shutil.copyfileobj(f, out)

    return {label: output_dir / f"{label}.txt" for label in labels}


if __name__ == "__main__":
    SOURCE = "cpp_bt"
    PERIOD = "quarter"

    split(PREPROCESSED_DIR / SOURCE, CORPUS_DIR / SOURCE / PERIOD, period=PERIOD)
