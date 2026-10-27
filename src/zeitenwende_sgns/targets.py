import polars as pl

from zeitenwende_sgns.config import TARGETS_PATH

NOUN_SUFFIXES = ["", "e", "en", "es", "n", "nen", "ns", "s"]
ADJECTIVE_SUFFIXES = ["", "e", "em", "en", "er", "es"]


def load_targets(path=TARGETS_PATH):
    """Read the target words, one lemma per line"""
    return path.read_text(encoding="utf-8").splitlines()


def inflections(target):
    """Guess regular inflected forms of a target word. Capitalized targets are taken
    as nouns, lowercase ones as adjectives (which may be capitalized at the start of
    a sentence). Umlaut plurals (Ausschüsse) are not covered."""
    if target[0].isupper():
        return [target + suffix for suffix in NOUN_SUFFIXES]
    forms = [target + suffix for suffix in ADJECTIVE_SUFFIXES]
    return forms + [form[0].upper() + form[1:] for form in forms]


def draft_overrides(preprocessed_dir, targets):
    """Find word forms that look like inflections of a target word but did not get
    the target as lemma. Review the result and save the correct rows as
    `LEMMA_OVERRIDES_PATH`."""
    candidates = pl.DataFrame(
        [(target, form) for target in targets for form in inflections(target)],
        schema=["target", "text"],
        orient="row",
    ).unique("text", keep="first")

    return (
        pl.scan_parquet(preprocessed_dir / "*.parquet")
        .select(pl.col("text_preprocessed").explode().explode().struct.unnest())
        .group_by("text", "lemma")
        .len("count")
        .join(candidates.lazy(), on="text")
        .filter(pl.col("lemma") != pl.col("target"))
        .select(
            pl.col("text").alias("form"),
            pl.col("target").alias("lemma"),
            pl.col("lemma").alias("current_lemma"),
            "count",
        )
        .sort("count", descending=True)
        .collect(engine="streaming")
    )
