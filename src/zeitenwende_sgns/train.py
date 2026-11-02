import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from gensim.models import KeyedVectors, Word2Vec
from tqdm import tqdm

from zeitenwende_sgns.config import CORPUS_DIR, MODELS_DIR

GENSIM_DOT_MESSAGE = [
    "Exception ignored in: ",
    "'gensim.models.word2vec_inner.our_dot_float'",
    "\n",
]


class GensimDotFilter:
    """Wraps stderr and drops the spurious "Exception ignored in: ...our_dot_float"
    messages of gensim (https://github.com/piskvorky/gensim/issues/3665), which
    CPython writes in three parts: prefix, object name, newline"""

    def __init__(self, stream):
        self.stream = stream
        self.pending = []

    def write(self, s):
        if not self.pending and s != GENSIM_DOT_MESSAGE[0]:
            return self.stream.write(s)
        self.pending.append(s)
        if self.pending != GENSIM_DOT_MESSAGE[: len(self.pending)]:
            self.stream.write("".join(self.pending))
            self.pending = []
        elif self.pending == GENSIM_DOT_MESSAGE:
            self.pending = []
        return len(s)

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def suppress_gensim_dot_warning() -> Generator[None]:
    stderr = sys.stderr
    sys.stderr = GensimDotFilter(stderr)
    try:
        yield
    finally:
        sys.stderr = stderr


def train(
    corpus_file: Path,
    output_path: Path,
    vector_size: int = 300,
    window: int = 4,
    negative: int = 5,
    sample: float = 1e-5,
    min_count: int = 5,
    epochs: int = 5,
    workers: int | None = None,
    seed: int = 1,
) -> KeyedVectors:
    """Train SGNS embeddings on a gensim `corpus_file` and save the word vectors to
    `output_path`. Defaults follow Hamilton et al. (2016). Runs with more than one
    worker are not exactly reproducible despite the seed."""
    with suppress_gensim_dot_warning():
        model = Word2Vec(
            corpus_file=str(corpus_file),
            sg=1,
            hs=0,
            vector_size=vector_size,
            window=window,
            negative=negative,
            sample=sample,
            min_count=min_count,
            epochs=epochs,
            workers=workers or os.cpu_count(),
            seed=seed,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    model.wv.save(str(tmp_path), separately=[])
    tmp_path.rename(output_path)
    return model.wv


def train_slices(corpus_dir: Path, output_dir: Path, **params) -> dict[str, Path]:
    """Train one model per corpus file in `corpus_dir` (as written by `split`) and
    save them as `<label>.kv` to `output_dir`. Existing models are skipped, so an
    interrupted run can simply be restarted. Returns the models by label."""
    models = {}
    for corpus_file in tqdm(sorted(corpus_dir.glob("*.txt")), unit="slice"):
        output_path = output_dir / f"{corpus_file.stem}.kv"
        if not output_path.exists():
            train(corpus_file, output_path, **params)
        models[corpus_file.stem] = output_path
    return models


if __name__ == "__main__":
    SOURCE = "cpp_bt"
    PERIOD = "quarter"

    train_slices(CORPUS_DIR / SOURCE / PERIOD, MODELS_DIR / SOURCE / PERIOD)
