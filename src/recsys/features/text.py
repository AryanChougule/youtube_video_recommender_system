"""Item text representation: video text -> dense, L2-normalised vectors.

Two interchangeable backends behind one interface:

``tfidf_svd`` (default)
    TF-IDF -> TruncatedSVD -> L2 normalise. This is classical Latent Semantic
    Analysis. It fixes TF-IDF's vocabulary-mismatch failure ("GPU" vs "graphics
    card" share no terms and would score 0 similarity) by projecting onto latent
    directions where co-occurring terms collapse together.

``sentence_transformers`` (optional)
    Neural sentence embeddings. Better at paraphrase, worse at rare technical
    terms, and it drags torch into the serving image -- which is why it is not
    the default. See docs/DESIGN_DECISIONS.md for the trade-off.

Design rule: **encode offline, serve with a dot product.** Vectors are
L2-normalised at build time, so cosine similarity at query time is a single
matrix multiply and the serving container needs nothing but NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class QueryEncoder(Protocol):
    """Anything that can turn free text into vectors in the item space."""

    def transform(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class TextIndex:
    """Item vectors plus the encoder needed to place a query in the same space."""

    vectors: np.ndarray          # (n_items, dims), L2-normalised, float32
    encoder: QueryEncoder
    backend: str
    dims: int
    meta: dict

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.asarray(self.encoder.transform(list(texts)), dtype=np.float32)
        if out.ndim == 1:
            out = out[None, :]
        return _l2_normalise(out)

    def similar_to_vector(self, query: np.ndarray, k: int = 20,
                          exclude: Sequence[int] = ()) -> tuple[np.ndarray, np.ndarray]:
        """Top-k most similar items to a query vector.

        Brute force is correct here: 6k items x 256 dims is ~1.5M multiply-adds,
        which numpy does in well under a millisecond. An approximate index
        (FAISS/HNSW) only starts paying for itself around 10^6 items, and it
        would cost us exactness plus a heavyweight dependency. Use the simple
        thing until the numbers say otherwise.
        """
        scores = self.vectors @ np.asarray(query, dtype=np.float32).ravel()
        if len(exclude):
            scores[np.asarray(list(exclude), dtype=int)] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return top, scores[top]


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _build_tfidf_svd(
    texts: Sequence[str], dims: int, max_features: int, ngram_max: int,
    seed: int, verbose: bool,
) -> tuple[np.ndarray, QueryEncoder, dict]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer

    # sublinear_tf: a term appearing 20x is not 20x more important than 1x.
    # min_df=2: a term in a single video cannot support similarity to anything.
    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, ngram_max), min_df=2,
        stop_words="english", sublinear_tf=True, strip_accents="unicode",
    )
    # dims must stay below the vocabulary size or SVD cannot factorise.
    probe = vectorizer.fit_transform(texts)
    dims = int(min(dims, max(2, min(probe.shape) - 1)))

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("svd", TruncatedSVD(n_components=dims, random_state=seed, algorithm="randomized")),
        ("norm", Normalizer(copy=False)),
    ])
    vectors = pipeline.fit_transform(texts).astype(np.float32)

    # The SVD component matrix is (dims x vocab) float64 and completely
    # dominates the serialised encoder -- 30MB of the 34MB artifact. Retrieval
    # scores agree to ~1e-6 in float32, which is far below the gap between
    # adjacent candidates, so the precision is dead weight in the container.
    svd = pipeline.named_steps["svd"]
    svd.components_ = svd.components_.astype(np.float32)

    explained = float(pipeline.named_steps["svd"].explained_variance_ratio_.sum())
    if verbose:
        print(f"  [text] tfidf_svd  vocab={len(vectorizer.vocabulary_):,}  "
              f"dims={dims}  explained_variance={explained:.1%}")
    meta = {
        "backend": "tfidf_svd", "dims": dims,
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "explained_variance": round(explained, 4),
    }
    return vectors, pipeline, meta


class _SentenceTransformerEncoder:
    """Picklable wrapper; loads the model lazily so unpickling stays cheap."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return self._load().encode(list(texts), convert_to_numpy=True,
                                   show_progress_bar=False, normalize_embeddings=True)

    def __getstate__(self) -> dict:
        return {"model_name": self.model_name, "_model": None}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


def _build_sentence_transformers(
    texts: Sequence[str], model_name: str, verbose: bool,
) -> tuple[np.ndarray, QueryEncoder, dict]:
    encoder = _SentenceTransformerEncoder(model_name)
    if verbose:
        print(f"  [text] sentence_transformers  model={model_name} (encoding {len(texts):,} docs)")
    vectors = np.asarray(encoder.transform(list(texts)), dtype=np.float32)
    meta = {"backend": "sentence_transformers", "dims": int(vectors.shape[1]),
            "model": model_name}
    return vectors, encoder, meta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_text_index(
    texts: Sequence[str],
    backend: str = "auto",
    dims: int = 256,
    max_features: int = 60000,
    ngram_max: int = 2,
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    seed: int = 42,
    verbose: bool = True,
) -> TextIndex:
    """Build item vectors with the requested backend.

    ``auto`` resolves to ``tfidf_svd``. That is deliberate, not lazy: on 6k
    short keyword-dense documents the neural gain is small, while it would add
    roughly 2GB to the deployed image and require torch at query time. Pass
    ``sentence_transformers`` explicitly to opt in.
    """
    texts = list(texts)
    resolved = "tfidf_svd" if backend == "auto" else backend
    if resolved not in ("tfidf_svd", "sentence_transformers"):
        raise ValueError(f"unknown text backend {backend!r}")

    if resolved == "sentence_transformers":
        try:
            vectors, encoder, meta = _build_sentence_transformers(texts, st_model, verbose)
        except ImportError:
            print("  [text] sentence-transformers not installed; falling back to tfidf_svd")
            resolved = "tfidf_svd"
    if resolved == "tfidf_svd":
        vectors, encoder, meta = _build_tfidf_svd(
            texts, dims, max_features, ngram_max, seed, verbose
        )

    vectors = _l2_normalise(vectors.astype(np.float32))
    meta["dims"] = int(vectors.shape[1])
    return TextIndex(vectors=vectors, encoder=encoder,
                     backend=meta["backend"], dims=meta["dims"], meta=meta)
