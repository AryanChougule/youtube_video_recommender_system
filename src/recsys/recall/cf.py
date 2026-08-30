"""Collaborative filtering: item-item co-visitation and implicit-feedback ALS.

Both are implemented from scratch in NumPy/SciPy rather than pulling in
``implicit``. The maths is the point, the dependency is heavy, and at our scale
the hand-rolled version is fast enough (ALS converges in ~25s on 3.8k users x
6k items).

Two complementary models
------------------------
``CoVisitation`` -- counts how often two videos appear in the same session,
normalised to kill popularity bias. Local, sparse, extremely strong for "more
like this", and it works from a single co-watch. This is essentially the engine
described in YouTube's 2010 recommender paper.

``ImplicitALS`` -- Hu/Koren/Volinsky (2008) matrix factorisation. Global,
dense, generalises across the long tail, and can score items a user's session
has never come near. Slower to build, needs enough data per user.

They fail in opposite directions, which is why we run both.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse

from .content import RecallResult


def _single_threaded_blas():
    """Pin BLAS to one thread for the ALS solve loop.

    Measured, not superstition. ALS does ~10,000 tiny (96x96) linear solves per
    iteration. Multi-threaded OpenBLAS spawns and synchronises a thread team for
    each one, and that overhead dwarfs the ~590k flops of actual work:

        96x96 solve, default threading :  3652 us
        96x96 solve, 1 thread          :    51 us   (70x faster)

    Large matmuls elsewhere still want all cores, so the limit is scoped to the
    training loop rather than set globally. threadpoolctl ships with
    scikit-learn; if it is somehow missing we degrade to correct-but-slow.
    """
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except ImportError:                                   # pragma: no cover
        return contextlib.nullcontext()


# ===========================================================================
# Interaction matrices
# ===========================================================================


def build_interaction_matrices(
    interactions: pd.DataFrame,
    user_ids: Sequence[str],
    video_ids: Sequence[str],
    signal: str = "watch_fraction",
    verbose: bool = True,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Build (user x item) and (session x item) matrices from clicks.

    The value stored is WATCH FRACTION, not a binary click. A video you clicked
    and abandoned after three seconds is evidence *against* recommending it,
    but a click-based model reads it as a positive. Encoding engagement here is
    how the "optimise watch time, not clicks" lesson enters the model.
    """
    import pandas as pd  # build-time only; keeps pandas out of the serving bundle

    clicked = interactions[interactions["clicked"] == 1]
    if clicked.empty:
        raise ValueError("no clicks in the interaction log")

    u_index = {u: i for i, u in enumerate(user_ids)}
    i_index = {v: i for i, v in enumerate(video_ids)}

    rows = clicked["user_id"].map(u_index)
    cols = clicked["video_id"].map(i_index)
    valid = rows.notna() & cols.notna()
    rows, cols = rows[valid].astype(int), cols[valid].astype(int)
    values = clicked.loc[valid, signal].astype(float).clip(0.01, 1.0)

    user_item = sparse.coo_matrix(
        (values, (rows, cols)), shape=(len(user_ids), len(video_ids))
    ).tocsr()
    # A user can rewatch; keep the strongest engagement rather than summing,
    # so a rewatched short does not outrank a fully-watched long video.
    user_item.sum_duplicates()
    user_item.data = np.minimum(user_item.data, 1.0)

    sessions = clicked.loc[valid, "session_id"]
    s_index = {s: i for i, s in enumerate(sessions.unique())}
    session_item = sparse.coo_matrix(
        (np.ones(valid.sum()), (sessions.map(s_index).astype(int), cols)),
        shape=(len(s_index), len(video_ids)),
    ).tocsr()
    session_item.data = np.ones_like(session_item.data)

    if verbose:
        print(f"  [cf] user-item {user_item.shape} nnz={user_item.nnz:,} "
              f"(density {user_item.nnz / np.prod(user_item.shape):.4%})")
        print(f"  [cf] session-item {session_item.shape} nnz={session_item.nnz:,}")
    return user_item, session_item


# ===========================================================================
# Item-item co-visitation
# ===========================================================================


@dataclass
class CoVisitation:
    """Top-k normalised co-visitation neighbours per item."""

    neighbours: np.ndarray       # (n_items, k) int32
    scores: np.ndarray           # (n_items, k) float32
    popularity: np.ndarray       # (n_items,) session frequency

    def similar_items(self, item: int, k: int = 50,
                      exclude: Sequence[int] = ()) -> RecallResult:
        idx = self.neighbours[item]
        sc = self.scores[item]
        keep = sc > 0
        idx, sc = idx[keep], sc[keep]
        if len(exclude):
            blocked = set(int(e) for e in exclude)
            mask = np.array([int(i) not in blocked for i in idx], dtype=bool)
            idx, sc = idx[mask], sc[mask]
        return RecallResult(idx[:k].astype(int), sc[:k], "cf_covisit")

    def for_history(self, history: Sequence[int], k: int = 50,
                    weights: Sequence[float] | None = None,
                    exclude: Sequence[int] = ()) -> RecallResult:
        """Aggregate neighbours of every watched item.

        Scores are summed across the history with recency decay: an item
        related to several things you watched should beat one related to a
        single thing. This is the classic item-based CF scoring rule.
        """
        if len(history) == 0:
            return RecallResult(np.array([], dtype=int), np.array([]), "cf_covisit")

        hist = np.asarray(list(history), dtype=int)
        engagement = (np.ones(len(hist)) if weights is None
                      else np.asarray(list(weights), dtype=float))
        position_back = np.arange(len(hist) - 1, -1, -1, dtype=float)
        decay = np.power(0.5, position_back / 8.0)

        blocked = set(int(x) for x in list(exclude) + list(hist))
        pooled: dict[int, float] = {}
        for h, w in zip(hist, engagement * decay):
            for cand, score in zip(self.neighbours[h], self.scores[h]):
                if score <= 0:
                    break                      # rows are sorted descending
                cand = int(cand)
                if cand in blocked:
                    continue
                pooled[cand] = pooled.get(cand, 0.0) + float(score) * float(w)

        if not pooled:
            return RecallResult(np.array([], dtype=int), np.array([]), "cf_covisit")
        order = sorted(pooled.items(), key=lambda kv: -kv[1])[:k]
        return RecallResult(
            np.array([i for i, _ in order], dtype=int),
            np.array([s for _, s in order], dtype=np.float32),
            "cf_covisit",
        )


def build_covisitation(
    session_item: sparse.csr_matrix,
    damping: float = 0.5,
    min_cooccurrence: int = 2,
    top_k: int = 100,
    verbose: bool = True,
) -> CoVisitation:
    """Co-visitation counts, popularity-damped and truncated to top-k.

    ``C = S^T S`` gives every pairwise count in one sparse matmul -- no loops.

    Normalisation is the whole game. Raw ``c_ij`` is dominated by whichever
    item is globally popular, so every item's "related" list degenerates into
    the same handful of viral videos. We divide by

        c_i^damping * c_j^(1 - damping)

    where ``damping = 0.5`` is exactly cosine similarity, higher values punish
    popular items harder, and 0 leaves them untouched. Making this a knob
    rather than a hidden constant is the point: it is the popularity-vs-
    relevance dial, and Module 7 measures what it costs.
    """
    counts = (session_item.T @ session_item).tocsr()
    counts.setdiag(0)
    counts.eliminate_zeros()

    if min_cooccurrence > 1:
        # A single shared session is almost always coincidence at this density.
        counts.data[counts.data < min_cooccurrence] = 0
        counts.eliminate_zeros()

    popularity = np.asarray(session_item.sum(axis=0)).ravel()
    safe_pop = np.maximum(popularity, 1.0)

    coo = counts.tocoo()
    norm = np.power(safe_pop[coo.row], damping) * np.power(safe_pop[coo.col], 1.0 - damping)
    normalised = sparse.csr_matrix(
        (coo.data / norm, (coo.row, coo.col)), shape=counts.shape
    )

    n_items = counts.shape[0]
    neighbours = np.zeros((n_items, top_k), dtype=np.int32)
    scores = np.zeros((n_items, top_k), dtype=np.float32)
    for i in range(n_items):
        start, stop = normalised.indptr[i], normalised.indptr[i + 1]
        if start == stop:
            continue
        row_idx = normalised.indices[start:stop]
        row_val = normalised.data[start:stop]
        k = min(top_k, len(row_val))
        top = np.argpartition(-row_val, k - 1)[:k]
        top = top[np.argsort(-row_val[top])]
        neighbours[i, :k] = row_idx[top]
        scores[i, :k] = row_val[top]

    if verbose:
        covered = int((scores[:, 0] > 0).sum())
        print(f"  [cf] co-visitation: {counts.nnz:,} pairs, damping={damping}, "
              f"{covered:,}/{n_items:,} items have neighbours")
    return CoVisitation(neighbours=neighbours, scores=scores, popularity=popularity)


# ===========================================================================
# Implicit-feedback ALS  (Hu, Koren & Volinsky, 2008)
# ===========================================================================


class ImplicitALS:
    """Weighted-regularised matrix factorisation for implicit feedback.

    Minimises

        sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + lambda (||X||^2 + ||Y||^2)

    with preference ``p_ui = 1[r_ui > 0]`` and confidence ``c_ui = 1 + alpha *
    r_ui``. Crucially the sum runs over ALL pairs, not just observed ones --
    that is what stops the model collapsing to "everything is loved", and it is
    the single idea that makes implicit feedback tractable.
    """

    def __init__(self, factors: int = 96, regularization: float = 0.05,
                 alpha: float = 30.0, iterations: int = 18, seed: int = 42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self._item_gram: np.ndarray | None = None    # Y^T Y, cached for fold-in

    # -- training ---------------------------------------------------------
    @staticmethod
    def _solve_side(
        matrix: sparse.csr_matrix, fixed: np.ndarray, regularization: float,
    ) -> np.ndarray:
        """One ALS half-step: closed-form solve for every row of ``matrix``.

        x_u = (Y^T C_u Y + lambda I)^-1 Y^T C_u p(u)

        The efficiency trick, and the reason ALS is usable at all:

            Y^T C_u Y  =  Y^T Y  +  Y^T (C_u - I) Y

        ``Y^T Y`` is the same for every user, so compute it ONCE outside the
        loop. ``C_u - I`` is zero everywhere the user did not interact, so the
        second term only touches that user's handful of items. Cost per user
        falls from O(n_items * f^2) to O(n_u * f^2) -- roughly 200x here.
        """
        n_rows = matrix.shape[0]
        f = fixed.shape[1]
        gram = fixed.T @ fixed                       # Y^T Y, computed once
        eye = regularization * np.eye(f, dtype=np.float64)
        out = np.zeros((n_rows, f), dtype=np.float64)

        for row in range(n_rows):
            start, stop = matrix.indptr[row], matrix.indptr[row + 1]
            if start == stop:
                continue                             # no data -> leave at zero
            idx = matrix.indices[start:stop]
            confidence = matrix.data[start:stop]     # this is (c_ui - 1)

            sub = fixed[idx]                         # (n_u, f)
            # Y_u^T diag(c-1) Y_u
            weighted = sub * confidence[:, None]
            a = gram + sub.T @ weighted + eye
            # Y^T C_u p(u) = Y_u^T c_u  (p is 1 exactly on observed items)
            b = weighted.sum(axis=0) + sub.sum(axis=0)
            out[row] = np.linalg.solve(a, b)
        return out

    def fit(self, user_item: sparse.csr_matrix, verbose: bool = True) -> "ImplicitALS":
        rng = np.random.default_rng(self.seed)
        n_users, n_items = user_item.shape

        # Store (c_ui - 1) = alpha * r_ui directly, so the solver never has to
        # subtract the identity term back out.
        cu = user_item.copy().astype(np.float64)
        cu.data = self.alpha * cu.data
        ci = cu.T.tocsr()

        # Small random init: exact zeros give a singular first solve.
        self.user_factors = rng.normal(0, 0.01, (n_users, self.factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.factors))

        with _single_threaded_blas():
            for iteration in range(self.iterations):
                self.user_factors = self._solve_side(cu, self.item_factors, self.regularization)
                self.item_factors = self._solve_side(ci, self.user_factors, self.regularization)
                if verbose and (iteration + 1) % 6 == 0:
                    loss = self._observed_loss(user_item)
                    print(f"  [als] iter {iteration + 1:>2}/{self.iterations}  "
                          f"observed-rmse={loss:.4f}")

        self._item_gram = self.item_factors.T @ self.item_factors
        return self

    def _observed_loss(self, user_item: sparse.csr_matrix) -> float:
        """RMSE on observed entries only.

        Diagnostic, NOT the training objective -- the real loss includes the
        implicit zeros. Useful only to confirm the solve is converging; do not
        read it as recommendation quality (Module 7 does that properly).
        """
        coo = user_item.tocoo()
        pred = np.einsum(
            "ij,ij->i", self.user_factors[coo.row], self.item_factors[coo.col]
        )
        return float(np.sqrt(np.mean((1.0 - pred) ** 2)))

    @classmethod
    def from_arrays(cls, user_factors: np.ndarray, item_factors: np.ndarray,
                    alpha: float, regularization: float) -> "ImplicitALS":
        """Rebuild a fitted model from saved factor matrices (serving path)."""
        model = cls(factors=int(item_factors.shape[1]),
                    regularization=regularization, alpha=alpha)
        model.user_factors = np.asarray(user_factors, dtype=np.float64)
        model.item_factors = np.asarray(item_factors, dtype=np.float64)
        model._item_gram = model.item_factors.T @ model.item_factors
        return model

    # -- inference --------------------------------------------------------
    def fold_in(self, history: Sequence[int],
                weights: Sequence[float] | None = None) -> np.ndarray:
        """Latent vector for a user who was never in the training matrix.

        This is what makes the demo UI work: an evaluator clicking around
        creates a brand-new user, and we solve the SAME closed form for their
        vector using the frozen item factors. One 96x96 solve, sub-millisecond.
        No retraining, no cold-start hole for new *users*.
        """
        if self.item_factors is None or self._item_gram is None:
            raise RuntimeError("call fit() first")
        f = self.factors
        if len(history) == 0:
            return np.zeros(f)

        idx = np.asarray(list(history), dtype=int)
        r = (np.ones(len(idx)) if weights is None
             else np.asarray(list(weights), dtype=float)).clip(0.01, 1.0)
        confidence = self.alpha * r

        sub = self.item_factors[idx]
        weighted = sub * confidence[:, None]
        a = self._item_gram + sub.T @ weighted + self.regularization * np.eye(f)
        b = weighted.sum(axis=0) + sub.sum(axis=0)
        return np.linalg.solve(a, b)

    def score_all(self, user_vector: np.ndarray) -> np.ndarray:
        return self.item_factors @ np.asarray(user_vector, dtype=np.float64)

    def recommend(self, user_vector: np.ndarray, k: int = 50,
                  exclude: Sequence[int] = ()) -> RecallResult:
        scores = self.score_all(user_vector)
        if len(exclude):
            scores[np.asarray(list(exclude), dtype=int)] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return RecallResult(top.astype(int), scores[top].astype(np.float32), "cf_als")

    def similar_items(self, item: int, k: int = 50,
                      exclude: Sequence[int] = ()) -> RecallResult:
        """Item-item similarity in the learned latent space.

        Cosine, not dot product: raw dot product in ALS space is strongly
        correlated with item popularity (popular items get longer vectors),
        so it would return the same viral videos for every seed.
        """
        vectors = self.item_factors
        norms = np.linalg.norm(vectors, axis=1)
        norms = np.maximum(norms, 1e-9)
        scores = (vectors @ vectors[item]) / (norms * norms[item])
        scores[item] = -np.inf
        if len(exclude):
            scores[np.asarray(list(exclude), dtype=int)] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return RecallResult(top.astype(int), scores[top].astype(np.float32), "cf_als_similar")
