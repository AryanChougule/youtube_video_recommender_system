"""The recommendation engine: Stage 1 -> Stage 2 -> Stage 3, plus explanations.

This is the only module the API talks to. It owns three things:

* **Orchestration.** Which recall sources run for which surface (home feed,
  watch page, search, cold start), fusion, ranking, policy.
* **Provenance.** Every item carries which sources proposed it, at what rank,
  and what the ranker thought. The brief asks evaluators to understand WHY a
  recommendation appeared; that is only possible if the pipeline records it as
  it goes rather than reconstructing a plausible story afterwards.
* **Safety of the served payload.** ``latent_quality`` is a hidden generative
  variable used by the simulator and by evaluation. It must never leave this
  process -- serving it would leak ground truth into the UI.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .artifacts import Artifacts
from .intent import DEFAULT_WINDOW, detect_intent
from .data.schema import split_tags
from .metrics import intra_list_diversity, novelty
from .policy.rerank import apply_policy
from .recall.blend import CandidateSet, reciprocal_rank_fusion

# Ordered by explanatory strength: a concrete "because you watched X" beats a
# vague "matches your taste", which beats "this is popular".
_SOURCE_PRIORITY = ["cf_covisit", "content_history", "cf_als", "channel", "trending", "popular"]


@dataclass
class RecommendedItem:
    video_id: str
    title: str
    channel_id: str
    channel_title: str
    category: str
    view_count: int
    like_count: int
    duration_seconds: int
    published_at: str
    age_days: float
    thumbnail_url: str
    rank: int
    score: float
    ranker_score: float
    sources: dict[str, int] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    explanation_detail: dict = field(default_factory=dict)
    policy_notes: list[str] = field(default_factory=list)


@dataclass
class RecommendationResponse:
    items: list[RecommendedItem]
    request: dict
    stages: dict
    diagnostics: dict

    def to_dict(self) -> dict:
        return {
            "items": [asdict(i) for i in self.items],
            "request": self.request,
            "stages": self.stages,
            "diagnostics": self.diagnostics,
        }


class RecommendationEngine:
    def __init__(self, artifacts: Artifacts):
        self.art = artifacts
        cfg = artifacts.config
        self.cfg = cfg
        catalog = artifacts.catalog
        self.channel_of = catalog["channel_id"].to_numpy()
        self.category_of = catalog["category"].to_numpy()
        self.titles = catalog["title"].to_numpy()
        self.channel_titles = catalog["channel_title"].to_numpy()
        self.age_days = artifacts.item_stats["age_days"].to_numpy(dtype=np.float64)
        # Exposure prior for novelty. Co-visitation popularity counts sessions,
        # which is a better proxy for "how often people actually meet this
        # video" than raw view_count.
        self.popularity = np.maximum(artifacts.covisitation.popularity, 0.0) + 1.0
        # Tags per item, for naming the session's current focus in the UI.
        self.tag_lists = [split_tags(t) for t in catalog["tags"].fillna("")]

    # ------------------------------------------------------------------
    # public surfaces
    # ------------------------------------------------------------------
    def recommend(
        self,
        history: Sequence[str] = (),
        watch_weights: Sequence[float] | None = None,
        seed_video: str | None = None,
        query: str | None = None,
        n: int | None = None,
        mmr_lambda: float | None = None,
        exploration_slots: int | None = None,
        max_per_channel: int | None = None,
        objective_weights: dict[str, float] | None = None,
        intent_alpha_scale: float | None = None,
        explain: bool = True,
    ) -> RecommendationResponse:
        cfg = self.cfg
        n = n or cfg.serving.default_n
        t_start = time.perf_counter()

        hist_idx = [self.art.idx(v) for v in history if v in self.art.video_index]
        weights = (list(watch_weights)[:len(hist_idx)] if watch_weights
                   else [1.0] * len(hist_idx))
        seed_idx = self.art.idx(seed_video) if seed_video in self.art.video_index else None
        exclude = list(hist_idx) + ([seed_idx] if seed_idx is not None else [])

        # ---- Stage 1: recall ----
        t0 = time.perf_counter()
        results = self._recall(hist_idx, weights, seed_idx, query, exclude)
        candidates = reciprocal_rank_fusion(
            results, weights=cfg.recall.weights, rrf_k=cfg.recall.rrf_k,
            max_candidates=cfg.recall.max_candidates,
        )
        t_recall = (time.perf_counter() - t0) * 1000

        if len(candidates) == 0:
            return RecommendationResponse(
                [], self._request_dict(history, seed_video, query, n),
                {"recall_ms": round(t_recall, 2)}, {"empty": True},
            )

        # ---- Stage 2: rank ----
        # On a watch page there is no browsing history, but there IS context:
        # the video being watched. Passing an empty history would zero every
        # match feature and leave the ranker scoring on item quality alone --
        # which discarded Stage 1's ordering entirely and put unrelated
        # categories above on-topic ones. Treat the seed as a 1-item history,
        # which is exactly what it is: the thing the user just watched.
        t0 = time.perf_counter()
        rank_history = hist_idx if hist_idx else (
            [seed_idx] if seed_idx is not None else []
        )
        rank_weights = weights if hist_idx else [1.0] * len(rank_history)
        ctx = self.art.features.build_context(rank_history, rank_weights)

        # Session intent. Used for EXPLANATION by default; blending it into the
        # query is off unless asked for, because it was measured not to help
        # (the profile's recency decay already acts as a session model). See
        # src/recsys/intent.py for the experiment.
        intent = detect_intent(
            self.art.text_index.vectors, rank_history, ctx.profile_vector,
            window=DEFAULT_WINDOW, tag_lists=self.tag_lists,
        )
        scale = (cfg.policy.intent_alpha_scale if intent_alpha_scale is None
                 else intent_alpha_scale)
        if scale > 0 and intent.detected and len(rank_history):
            blended = (scale * intent.alpha) * intent.session_vector +                       (1.0 - scale * intent.alpha) * ctx.profile_vector
            norm = np.linalg.norm(blended)
            if norm > 1e-9:
                ctx.profile_vector = (blended / norm).astype(np.float32)

        ranker_scores = self._rank(ctx, candidates, objective_weights)
        t_rank = (time.perf_counter() - t0) * 1000

        # ---- Stage 3: policy ----
        t0 = time.perf_counter()
        policy = apply_policy(
            candidates=candidates.indices,
            ranker_scores=ranker_scores,
            vectors=self.art.text_index.vectors,
            age_days=self.age_days[candidates.indices],
            channel_of=self.channel_of[candidates.indices],
            category_of=self.category_of[candidates.indices],
            popularity=self.popularity,
            k=n,
            mmr_lambda=cfg.policy.mmr_lambda if mmr_lambda is None else mmr_lambda,
            max_per_channel=(cfg.policy.max_per_channel if max_per_channel is None
                             else max_per_channel),
            freshness_halflife=cfg.policy.freshness_halflife_days,
            freshness_weight=cfg.policy.freshness_weight,
            exploration_slots=(cfg.policy.exploration_slots if exploration_slots is None
                               else exploration_slots),
            seed=cfg.project.seed,
        )
        t_policy = (time.perf_counter() - t0) * 1000

        score_lookup = {int(i): float(s) for i, s in zip(candidates.indices, ranker_scores)}

        # Per-objective breakdown for the items that actually made the page.
        # Computed on the final ~24 rather than all ~400 candidates: the panel
        # only ever shows what shipped, and it keeps the cost negligible.
        objective_breakdown: dict[int, dict] = {}
        if self.art.multitask is not None and len(policy.order):
            final_matrix = self.art.features.build(ctx, policy.order)
            rows = self.art.multitask.explain_scores(final_matrix, objective_weights)
            objective_breakdown = {int(i): r for i, r in zip(policy.order, rows)}

        mode = self._mode(history, seed_video, query)
        items = self._materialise(policy, candidates, score_lookup,
                                  rank_history, explain, mode, seed_idx,
                                  objective_breakdown, objective_weights)

        total_ms = (time.perf_counter() - t_start) * 1000
        return RecommendationResponse(
            items=items,
            request=self._request_dict(history, seed_video, query, n),
            stages={
                "recall_ms": round(t_recall, 2),
                "rank_ms": round(t_rank, 2),
                "policy_ms": round(t_policy, 2),
                "total_ms": round(total_ms, 2),
                "sources": candidates.source_sizes,
                "n_candidates": len(candidates),
                "catalog_size": self.art.n_items,
            },
            diagnostics={**self._diagnostics(policy.order),
                         "session_intent": intent.to_dict(),
                         "intent_applied": round(scale, 3)},
        )

    def similar(self, video_id: str, n: int = 12) -> RecommendationResponse:
        """Watch-page rail: 'more like this', with no user personalisation."""
        return self.recommend(seed_video=video_id, n=n)

    def search(self, query: str, n: int = 24,
               history: Sequence[str] = ()) -> RecommendationResponse:
        return self.recommend(query=query, history=history, n=n)

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------
    def _recall(self, hist_idx, weights, seed_idx, query, exclude) -> dict:
        cfg = self.cfg
        art = self.art
        results: dict = {}

        if query:
            # Search is intent-dominant: the query is a far stronger statement
            # of what someone wants right now than their long-run history, so
            # it leads and personalisation only re-ranks around it.
            results["content_history"] = art.content.search(
                query, k=cfg.recall.content_k, exclude=exclude
            )
        elif seed_idx is not None:
            results["content_history"] = art.content.similar_items(
                seed_idx, k=cfg.recall.content_k, exclude=exclude
            )
            results["cf_covisit"] = art.covisitation.similar_items(
                seed_idx, k=cfg.recall.cf_item_k, exclude=exclude
            )
            results["cf_als"] = art.als.similar_items(
                seed_idx, k=cfg.recall.als_k, exclude=exclude
            )
        elif hist_idx:
            results["content_history"] = art.content.for_history(
                hist_idx, k=cfg.recall.content_k, weights=weights, exclude=exclude
            )
            results["cf_covisit"] = art.covisitation.for_history(
                hist_idx, k=cfg.recall.cf_item_k, weights=weights, exclude=exclude
            )
            user_vector = art.als.fold_in(hist_idx, weights)
            results["cf_als"] = art.als.recommend(
                user_vector, k=cfg.recall.als_k, exclude=exclude
            )
            results["channel"] = art.channel.for_history(
                hist_idx, k=cfg.recall.channel_k, weights=weights, exclude=exclude
            )

        # Trending always runs. For a cold user it is the entire answer; for a
        # warm one it is a small floor of fresh material that personalisation
        # would otherwise never surface, because nothing in the history points
        # at a video uploaded this morning.
        results["trending"] = art.trending.trending(k=cfg.recall.trending_k, exclude=exclude)
        if not hist_idx and seed_idx is None and not query:
            results["popular"] = art.trending.popular(k=cfg.recall.trending_k, exclude=exclude)
        return results

    def _rank(self, ctx, candidates: CandidateSet,
              objective_weights: dict[str, float] | None = None) -> np.ndarray:
        # Multi-objective path: the heads are fixed at training time but the
        # OBJECTIVE is chosen per request, which is what lets the UI turn the
        # system from engagement-maximising to satisfaction-maximising without
        # retraining anything.
        if objective_weights and self.art.multitask is not None:
            matrix = self.art.features.build(ctx, candidates.indices)
            return self.art.multitask.score(matrix, weights=objective_weights)
        if self.art.ranker is None:
            # Graceful degradation: without a ranker, fused recall order is a
            # perfectly serviceable ranking. The system should not 500 because
            # one artifact is missing.
            return candidates.fused_scores.astype(np.float32)
        matrix = self.art.features.build(ctx, candidates.indices)
        return self.art.ranker.score(matrix)

    # ------------------------------------------------------------------
    # presentation
    # ------------------------------------------------------------------
    def _materialise(self, policy, candidates, score_lookup, hist_idx,
                     explain: bool, mode: str = "personalised",
                     seed_idx: int | None = None,
                     objective_breakdown: dict | None = None,
                     objective_weights: dict | None = None) -> list[RecommendedItem]:
        catalog = self.art.catalog
        items: list[RecommendedItem] = []
        for rank, item_idx in enumerate(policy.order):
            item_idx = int(item_idx)
            row = catalog.iloc[item_idx]
            sources = candidates.sources_for(item_idx)
            source_scores = {
                name: round(candidates.score_from(name, item_idx), 4)
                for name in sources
            }
            notes = policy.policy_notes.get(item_idx, [])
            explanation, detail = (
                self._explain(item_idx, sources, source_scores, hist_idx, notes,
                              mode, seed_idx)
                if explain else ("", {})
            )
            if explain and objective_breakdown and item_idx in objective_breakdown:
                breakdown = dict(objective_breakdown[item_idx])
                probabilities = breakdown.pop("_probabilities", {})
                detail["objectives"] = {
                    "probabilities": probabilities,
                    "contributions": breakdown,
                    "weights": {
                        k: round(float(v), 4) for k, v in
                        (self.art.multitask.weights | (
                            {kk: float(vv) for kk, vv in (objective_weights or {}).items()}
                        )).items()
                    },
                    "total": round(float(sum(breakdown.values())), 4),
                }
            items.append(RecommendedItem(
                video_id=str(row["video_id"]),
                title=str(row["title"]),
                channel_id=str(row["channel_id"]),
                channel_title=str(row["channel_title"]),
                category=str(row["category"]),
                view_count=int(row["view_count"]),
                like_count=int(row["like_count"]),
                duration_seconds=int(row["duration_seconds"]),
                published_at=pd.Timestamp(row["published_at"]).strftime("%Y-%m-%d"),
                age_days=round(float(self.age_days[item_idx]), 1),
                thumbnail_url=str(row["thumbnail_url"]),
                rank=rank,
                score=round(float(policy.base_scores[rank]), 4),
                ranker_score=round(score_lookup.get(item_idx, 0.0), 4),
                sources=sources,
                source_scores=source_scores,
                explanation=explanation,
                explanation_detail=detail,
                policy_notes=notes,
            ))
        return items

    def _explain(self, item_idx: int, sources: dict[str, int],
                 source_scores: dict[str, float], hist_idx: list[int],
                 notes: list[str], mode: str = "personalised",
                 seed_idx: int | None = None) -> tuple[str, dict]:
        """Human-readable reason plus the structured evidence behind it.

        The sentence names the strongest source that actually contributed --
        ``sources`` is recorded during fusion, so this is a faithful report of
        why the item survived Stage 1, not a plausible story invented after.

        Wording is MODE-AWARE, and that is a correctness issue rather than
        polish. On a watch page there is no user profile at all, so "viewers
        with a taste profile like yours" would be a claim about something that
        does not exist; in search, the query is the reason, not the history.
        An explanation that misdescribes its own evidence is worse than none.
        """
        detail = {
            "sources": sources,
            "source_scores": source_scores,
            "policy": notes,
            "mode": mode,
        }
        if "exploration" in notes:
            return "Something different - picked to widen your feed", detail

        leader = next((s for s in _SOURCE_PRIORITY if s in sources), None)

        # ---- watch page: the seed video is the only context ----
        if mode == "watch_page" and seed_idx is not None:
            detail["because_of_video"] = str(self.art.video_ids[seed_idx])
            seed_title = self._short(self.titles[seed_idx])
            if leader == "cf_covisit":
                return f'People who watched "{seed_title}" also watched this', detail
            if leader == "cf_als":
                return f'Watched by the same viewers as "{seed_title}"', detail
            if leader == "channel":
                return f'More from "{self.channel_titles[item_idx]}"', detail
            if leader in ("trending", "popular"):
                return f"Trending in {self.category_of[item_idx]}", detail
            return f'Similar to "{seed_title}"', detail

        # ---- search: the query is the reason ----
        if mode == "search":
            if leader in ("trending", "popular"):
                return f"Trending in {self.category_of[item_idx]}", detail
            return f"Matches your search in {self.category_of[item_idx]}", detail

        # ---- personalised feed ----
        if leader in ("cf_covisit", "content_history") and hist_idx:
            anchor, similarity = self.art.content.nearest_reason(item_idx, hist_idx)
            if anchor >= 0:
                detail["because_of_video"] = str(self.art.video_ids[anchor])
                detail["anchor_similarity"] = round(similarity, 3)
                verb = ("People who watched" if leader == "cf_covisit" else "Because you watched")
                tail = (" also watched this" if leader == "cf_covisit" else "")
                return f'{verb} "{self._short(self.titles[anchor])}"{tail}', detail

        if leader == "cf_als":
            return "Viewers with a taste profile like yours watch this", detail
        if leader == "channel":
            return f'More from "{self.channel_titles[item_idx]}"', detail
        if leader in ("trending", "popular"):
            return f"Trending in {self.category_of[item_idx]}", detail
        if leader == "content_history":
            return "Matches what you have been watching", detail
        return f"Popular in {self.category_of[item_idx]}", detail

    @staticmethod
    def _mode(history, seed_video, query) -> str:
        if query:
            return "search"
        if seed_video:
            return "watch_page"
        return "personalised" if len(history) else "cold_start"

    @staticmethod
    def _short(title: str, limit: int = 42) -> str:
        title = str(title)
        return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"

    def _diagnostics(self, order: np.ndarray) -> dict:
        """Per-response beyond-accuracy stats, surfaced in the UI debug panel."""
        idx = np.asarray(order, dtype=int)
        if len(idx) == 0:
            return {}
        categories = self.category_of[idx]
        channels = self.channel_of[idx]
        return {
            "intra_list_diversity": round(
                intra_list_diversity(idx, self.art.text_index.vectors), 4),
            "novelty_bits": round(novelty(idx, self.popularity), 2),
            "distinct_categories": int(len(set(categories))),
            "distinct_channels": int(len(set(channels))),
            "category_mix": {str(c): int((categories == c).sum())
                             for c in dict.fromkeys(categories)},
            "median_age_days": round(float(np.median(self.age_days[idx])), 1),
        }

    def _request_dict(self, history, seed_video, query, n) -> dict:
        return {
            "history": list(history), "seed_video": seed_video,
            "query": query, "n": n,
            "mode": self._mode(history, seed_video, query),
        }
