"""A NumPy re-implementation of the TF-IDF -> SVD -> L2 query encoder.

Same reasoning as :mod:`recsys.serving.trees`: fitting the encoder is the hard
part, and none of that machinery is needed to *use* it. What a fitted
``TfidfVectorizer -> TruncatedSVD -> Normalizer`` pipeline actually does to a
query string is:

    1. lowercase, strip accents
    2. split on the word-token pattern
    3. drop English stop words
    4. form unigrams and bigrams (bigrams AFTER stop-word removal)
    5. count, apply sublinear tf (1 + log tf), multiply by the fitted idf
    6. L2 normalise, project through the SVD components, L2 normalise again

Steps 1-4 are string handling; steps 5-6 are one sparse lookup and one dense
matrix multiply. All of it fits in a hundred lines of NumPy, which is what
lets the serving container drop scikit-learn and SciPy entirely.

The fitted state is three things -- the vocabulary, the idf vector, and the SVD
components -- plus the stop-word list, which is exported alongside them rather
than re-imported from scikit-learn at runtime. Exporting the list is the point:
a serving path that had to import ``sklearn.feature_extraction.text`` to get a
frozenset of 318 English words would not be dependency-free at all.

Reproducing the tokenizer by hand is the risky part of this file, so
``scripts/12_export_serving.py`` asserts the exported encoder matches
scikit-learn's output on every catalog document, not on a handful of examples.
"""

from __future__ import annotations

import json
import re
import unicodedata

import numpy as np

#: scikit-learn's default ``token_pattern``. Two-or-more word characters, so
#: single letters are dropped -- which is why "a" and "I" never appear in the
#: vocabulary regardless of the stop-word list.
TOKEN_PATTERN = r"(?u)\b\w\w+\b"

_TOKENIZER = re.compile(TOKEN_PATTERN)


def strip_accents_unicode(text: str) -> str:
    """Port of ``sklearn.feature_extraction.text.strip_accents_unicode``.

    The ASCII fast path is not an optimisation we invented -- it is part of the
    original, and it matters for equivalence: NFKD normalisation can alter
    strings that were already accent-free, so skipping it for pure-ASCII input
    is load-bearing, not just faster.
    """
    try:
        text.encode("ASCII", errors="strict")
        return text
    except UnicodeEncodeError:
        normalised = unicodedata.normalize("NFKD", text)
        return "".join(c for c in normalised if not unicodedata.combining(c))


class NumpyTfidfSvd:
    """Query encoder: text in, dense L2-normalised vector out."""

    __slots__ = ("vocabulary", "idf", "components", "stop_words", "ngram_max",
                 "sublinear_tf")

    def __init__(self, vocabulary: dict, idf, components, stop_words,
                 ngram_max: int = 2, sublinear_tf: bool = True):
        self.vocabulary = vocabulary
        self.idf = np.asarray(idf, dtype=np.float64)
        self.components = np.asarray(components, dtype=np.float32)
        self.stop_words = frozenset(stop_words)
        self.ngram_max = int(ngram_max)
        self.sublinear_tf = bool(sublinear_tf)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_pipeline(cls, pipeline) -> "NumpyTfidfSvd":
        """Lift the fitted state out of a scikit-learn Pipeline. Build time only."""
        tfidf = pipeline.named_steps["tfidf"]
        svd = pipeline.named_steps["svd"]
        if tfidf.token_pattern != TOKEN_PATTERN or tfidf.analyzer != "word":
            raise ValueError("the NumPy encoder assumes the default word analyzer")
        if tfidf.strip_accents != "unicode" or not tfidf.lowercase:
            raise ValueError("the NumPy encoder assumes lowercase + unicode accents")
        if tfidf.norm != "l2" or tfidf.binary:
            raise ValueError("the NumPy encoder assumes l2 norm and non-binary counts")
        if tfidf.ngram_range[0] != 1:
            raise ValueError("the NumPy encoder assumes ngram_range starts at 1")
        return cls(
            vocabulary={str(k): int(v) for k, v in tfidf.vocabulary_.items()},
            idf=tfidf.idf_,
            components=svd.components_,
            stop_words=tfidf.get_stop_words() or (),
            ngram_max=int(tfidf.ngram_range[1]),
            sublinear_tf=bool(tfidf.sublinear_tf),
        )

    # -- the analyzer -----------------------------------------------------
    def analyze(self, doc: str) -> list:
        """Tokens and n-grams, matching scikit-learn's word analyzer exactly."""
        tokens = _TOKENIZER.findall(strip_accents_unicode(doc.lower()))
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
        if self.ngram_max == 1:
            return tokens
        # Bigrams are built from the stop-word-FILTERED unigrams, which is what
        # scikit-learn does. Building them first would produce different terms
        # and silently miss most of the vocabulary.
        out = list(tokens)
        n_tokens = len(tokens)
        for n in range(2, min(self.ngram_max + 1, n_tokens + 1)):
            for i in range(n_tokens - n + 1):
                out.append(" ".join(tokens[i:i + n]))
        return out

    # -- inference --------------------------------------------------------
    def transform(self, texts) -> np.ndarray:
        texts = list(texts)
        n_vocab = len(self.idf)
        tfidf = np.zeros((len(texts), n_vocab), dtype=np.float64)
        for row, doc in enumerate(texts):
            counts: dict = {}
            for term in self.analyze(str(doc)):
                col = self.vocabulary.get(term)
                if col is not None:
                    counts[col] = counts.get(col, 0) + 1
            if not counts:
                continue
            cols = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
            vals = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
            if self.sublinear_tf:
                vals = 1.0 + np.log(vals)
            tfidf[row, cols] = vals * self.idf[cols]

        # The dense (n_queries x vocab) intermediate is fine because queries
        # arrive one or a few at a time. Encoding the whole catalog this way
        # would not be -- that happens at build time, in scikit-learn.
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        tfidf /= np.maximum(norms, 1e-12)
        return (tfidf.astype(np.float32) @ self.components.T).astype(np.float32)

    # -- persistence ------------------------------------------------------
    def arrays(self) -> dict:
        return {"text__idf": self.idf, "text__components": self.components}

    def meta(self) -> dict:
        return {
            "vocabulary": self.vocabulary,
            "stop_words": sorted(self.stop_words),
            "ngram_max": self.ngram_max,
            "sublinear_tf": self.sublinear_tf,
        }

    @classmethod
    def from_arrays(cls, blob, meta: dict) -> "NumpyTfidfSvd":
        return cls(vocabulary=meta["vocabulary"], idf=blob["text__idf"],
                   components=blob["text__components"],
                   stop_words=meta["stop_words"], ngram_max=meta["ngram_max"],
                   sublinear_tf=meta["sublinear_tf"])

    def __repr__(self) -> str:
        return (f"NumpyTfidfSvd(vocab={len(self.vocabulary):,}, "
                f"dims={self.components.shape[0]})")


def dumps_meta(meta: dict) -> str:
    return json.dumps(meta, separators=(",", ":"))
