"""Session intent: what this person wants *right now*, not what they like.

The problem
-----------
A recommender built on a long-term profile answers "what does this user
generally enjoy?". That is the wrong question surprisingly often. Someone whose
profile says AI / cooking / gaming may today be working through one specific
thing -- learning RAG before an interview tomorrow. Filling their feed with
assorted AI videos because the profile says "AI" is a failure, even though every
item is defensible in isolation.

It is also the mechanism behind two failures measured elsewhere in this project:
a single-interest history collapsing into a monoculture, and a multi-interest
history producing a centroid that matches nothing.

The model
---------
Split the watch history in two:

    long-term profile   recency-weighted mean over the whole history
    session vector      mean over the last ``window`` watches

Then blend them, with the weight decided by how *coherent* the session is:

    query = normalise( alpha * session + (1 - alpha) * long_term )

**Coherence** is the mean pairwise cosine similarity of the session items. Four
consecutive videos about vector databases are tightly clustered; four unrelated
videos are not. High coherence is evidence of deliberate intent; low coherence
means the person is browsing, and the stable profile is the better guide.

**Novelty** is 1 - cos(session, long_term). A session that is both coherent AND
unlike the profile is the strongest possible signal of a deliberate goal --
precisely the "learning RAG for an interview" case -- so novelty raises alpha
further. A coherent session that merely repeats the profile does not need the
boost, because both vectors already agree.

DOES IT ACTUALLY IMPROVE RANKING?  No -- and the reason is worth more
---------------------------------------------------------------------
This was tested properly before being believed, and the answer is a clean
negative. Across both counterfactually-valid protocols, every gating rule and
every blend weight, an explicit session vector fails to beat the plain profile:

    Protocol A (re-rank logged impressions)   best gate:  -0.1%
    Protocol B (1 positive vs 100 negatives)  best gate:  +0.1%
    alpha sweep 0.0 -> 1.0                    best alpha:  0.0

The diagnosis is the valuable part. The profile is a RECENCY-WEIGHTED mean with
a half-life of 8 positions, so it is already dominated by recent watches:

    cos(recency-decayed profile, session vector) = 0.803

Exponential recency decay is already a soft session model. The decisive test is
to remove it -- on a uniform-mean profile with no decay, session blending
suddenly works:

    profile basis                    alpha=0   alpha=0.5    lift
    recency-decayed (our baseline)    0.4522      0.4395    -2.8%
    uniform mean (no decay)           0.4433      0.4547    +2.6%

**Session-intent blending helps exactly when your profile lacks recency
weighting. Ours has it. The mechanism was already implemented under a different
name.** A half-life sweep (2 -> infinity) confirms the curve is flat near 8, so
there is no tuning win hiding here either.

The general lesson, which generalises well past recommenders: before adding a
mechanism, check whether a simpler one already in the system covers it. A
plausible product story is not evidence.

So what IS shipped
------------------
Intent DETECTION ships and is used for **explanation**, not ranking: naming the
session's current focus ("sourdough starter, proofing") makes the feed legible
in a way the raw profile cannot. Intent BLENDING ships switched off
(``policy.intent_alpha_scale: 0.0``) and is exposed as a slider in the
Recommendation Lab, so an evaluator can turn it on and watch it fail to help --
which is a more useful demonstration than hiding it.

How well can the DETECTOR possibly work?
----------------------------------------
Measured against the simulator's ground-truth session intent, coherence
separates focused from browsing sessions at AUC 0.6165. That sounds weak until
you compute the ceiling the same way the rest of this project does -- by
running the identical detector on the TRUE latent topic mixtures:

    text vectors (what we serve)   AUC 0.6165
    ALS latent (behavioural)       AUC 0.5766
    ORACLE: true topic mixtures    AUC 0.6251   <- the ceiling

The servable detector reaches 98.6% of the oracle. **The bottleneck is not the
item representation -- it is that five clicks simply do not contain much
information about intent.** A fancier session encoder would be optimising
against a ceiling that is already nearly reached, which is why P2 in
docs/FUTURE_WORK.md is ranked below work with more headroom.

Note also that binary detection is not the goal. Alpha is continuous, so the
blend degrades gracefully: a weak signal nudges the query, a strong one moves
it. What matters is the ranking delta, measured in scripts/10_evaluate_intent.py.

Deliberately simple
-------------------
No model is trained here. Intent is computed from geometry the retrieval layer
already has, which means it costs ~50 microseconds, has no training data
requirement, cannot drift out of sync with the item vectors, and is completely
inspectable -- ``SessionIntent`` carries the numbers that produced it so the UI
can show its reasoning. A learned session encoder (GRU4Rec / SASRec) is the
principled upgrade and is P2 in docs/FUTURE_WORK.md; this is the version that
can be shipped and audited today.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# How many of the most recent watches count as "this session".
DEFAULT_WINDOW = 5

# Coherence below LO is treated as browsing, above HI as a deliberate focus.
# CALIBRATED against the observed distribution, not guessed: p25 and p90 of
# session coherence in the text space (0.069 / 0.231). The first version used
# 0.18/0.55 -- plausible-looking numbers that sat above the 90th percentile, so
# alpha was ~0 for nearly every session and the detector did nothing at all.
# Re-derive with `python scripts/10_evaluate_intent.py`, which prints them.
COHERENCE_LO = 0.07
COHERENCE_HI = 0.23

# Cap on how far the session may pull the query away from the long-term
# profile. Never 1.0: a user is more than their last five clicks, and letting a
# single stray watch fully hijack the feed is a worse failure than being a
# little slow to adapt.
MAX_ALPHA = 0.80

# How much genuine novelty (session unlike the profile) adds on top.
NOVELTY_BOOST = 0.25


@dataclass
class SessionIntent:
    """A detected intent, plus every number that produced it."""

    vector: np.ndarray                  # the blended query vector
    session_vector: np.ndarray
    profile_vector: np.ndarray
    alpha: float                        # weight given to the session
    coherence: float                    # 0..1, how focused the session is
    novelty: float                      # 0..1, how unlike the profile it is
    focus_items: list[int] = field(default_factory=list)
    label: str = ""                     # human-readable, for the UI
    detected: bool = False              # did we find a usable intent at all

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "label": self.label,
            "alpha": round(self.alpha, 3),
            "coherence": round(self.coherence, 3),
            "novelty": round(self.novelty, 3),
            "window": len(self.focus_items),
        }


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)


def session_coherence(vectors: np.ndarray) -> float:
    """Mean pairwise cosine similarity of the session's item vectors.

    Vectors are already L2-normalised, so the Gram matrix is cosine directly.
    One item cannot be coherent or incoherent, so it returns 0 -- which makes
    the blend fall back to the profile, the right behaviour when there is no
    session to speak of.
    """
    if len(vectors) < 2:
        return 0.0
    gram = vectors @ vectors.T
    n = len(vectors)
    off_diagonal = (gram.sum() - np.trace(gram)) / (n * (n - 1))
    return float(np.clip(off_diagonal, 0.0, 1.0))


def label_from_tags(tag_lists: Sequence[Sequence[str]], top_k: int = 3) -> str:
    """Name the intent from the tags shared across the session's videos.

    Tags carry topic with far less stylistic noise than titles -- the same
    finding that drove the text-index field weighting in
    ``recsys.data.catalog.catalog_text``.
    """
    counts: Counter[str] = Counter()
    for tags in tag_lists:
        counts.update({t for t in tags if t})
    if not counts:
        return ""
    # A tag shared by several videos is far more likely to be the point of the
    # session than one that appears once.
    shared = [t for t, c in counts.most_common() if c > 1]
    chosen = (shared or [t for t, _ in counts.most_common()])[:top_k]
    return ", ".join(chosen)


def detect_intent(
    item_vectors: np.ndarray,
    history: Sequence[int],
    profile_vector: np.ndarray,
    window: int = DEFAULT_WINDOW,
    tag_lists: Sequence[Sequence[str]] | None = None,
    max_alpha: float = MAX_ALPHA,
) -> SessionIntent:
    """Detect the current session's intent and build the query vector."""
    hist = np.asarray(list(history), dtype=int)
    dims = item_vectors.shape[1]

    if len(hist) == 0:
        zero = np.zeros(dims, dtype=np.float32)
        return SessionIntent(vector=zero, session_vector=zero,
                             profile_vector=zero, alpha=0.0,
                             coherence=0.0, novelty=0.0)

    recent = hist[-window:]
    recent_vectors = item_vectors[recent]
    session_vector = _l2(recent_vectors.mean(axis=0))

    coherence = session_coherence(recent_vectors)
    novelty = float(np.clip(1.0 - float(session_vector @ profile_vector), 0.0, 1.0))

    # Map coherence onto [0, 1], then let novelty push it further. A session
    # that is BOTH focused and unlike the profile is the clearest evidence of a
    # deliberate goal.
    focus = float(np.clip(
        (coherence - COHERENCE_LO) / (COHERENCE_HI - COHERENCE_LO), 0.0, 1.0))
    alpha = float(np.clip(focus * (1.0 + NOVELTY_BOOST * novelty), 0.0, 1.0) * max_alpha)

    blended = _l2(alpha * session_vector + (1.0 - alpha) * profile_vector)

    label = ""
    if tag_lists is not None:
        label = label_from_tags([tag_lists[int(i)] for i in recent])

    return SessionIntent(
        vector=blended,
        session_vector=session_vector,
        profile_vector=profile_vector,
        alpha=alpha,
        coherence=coherence,
        novelty=novelty,
        focus_items=[int(i) for i in recent],
        label=label,
        # "Detected" means the session is focused enough to actually move the
        # query. Reporting an intent that changes nothing would be theatre.
        detected=alpha > 0.15,
    )
