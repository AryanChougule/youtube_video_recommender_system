"""Guards for the evaluation harness itself.

An evaluation bug is more expensive than a model bug. A model that ranks badly
shows up in the metrics; a harness that measures the wrong thing produces
confident numbers that are simply false, and nothing downstream disagrees with
them.

Both tests here are regressions for that exact failure. When the serving
catalog was made pandas-free, hidden generative variables were deliberately
excluded from it -- correct, because a model able to see them would be graded
against its own answer key. But three evaluation call sites were still reading
ground truth from that catalog. Two began raising KeyError. The third had an

    latent_clickbait if present else np.zeros(len(catalog))

fallback and reported **clickbait@1 = 0.0000 for every strategy** -- a result
that looks like a finding ("multi-objective eliminated clickbait!") and is an
absence. It would have been quoted in the docs as a headline number.
"""

from __future__ import annotations

import numpy as np
import pytest

from recsys.artifacts import load_artifacts
from recsys.engine import RecommendationEngine
from recsys.groundtruth import GroundTruthUnavailable, load_latent


def test_hidden_variables_are_absent_from_the_serving_catalog():
    """The exclusion is the safety property; assert it rather than assume it."""
    catalog = load_artifacts().catalog
    assert not [c for c in catalog.columns if c.startswith("latent_")]
    for name in ("latent_quality", "latent_clickbait"):
        with pytest.raises(KeyError):
            catalog[name]


def test_ground_truth_loads_from_the_build_output():
    """...and is still reachable where it actually lives, for evaluation."""
    n = len(load_artifacts().catalog)
    for name in ("latent_quality", "latent_clickbait"):
        values = load_latent(name, expected_rows=n)
        assert len(values) == n
        # A zero-filled fallback would pass a length check, so assert the
        # values carry information -- that is the bug this guards.
        assert values.std() > 0, f"{name} is constant; ground truth is not loaded"


def test_missing_ground_truth_raises_instead_of_defaulting():
    """The whole point: absence must stop an evaluation, not silently pass."""
    with pytest.raises(GroundTruthUnavailable):
        load_latent("latent_does_not_exist")
    with pytest.raises(GroundTruthUnavailable):
        load_latent("latent_quality", expected_rows=7)


def test_recall_reports_whether_a_query_matched():
    """Pins ``_recall``'s contract, which three scripts unpack directly.

    It returns ``(results, query_matched)``. When that changed, two scripts
    kept treating the tuple as a dict and failed only when someone ran the
    full evaluation -- long after the change looked fine.
    """
    engine = RecommendationEngine(load_artifacts())
    results, matched = engine._recall([], [], None, None, [])
    assert isinstance(results, dict) and matched is True
    assert "trending" in results

    _, matched = engine._recall([], [], None, "machine learning", [])
    assert matched is False, "a fully out-of-vocabulary query is not a match"


def test_oracle_scores_differ_across_items():
    """The oracle control underpins the headline evaluation finding.

    It is built from the simulator's own generative parameters, so if it ever
    silently degrades to a constant the whole "is this metric even measuring
    anything?" argument in EVALUATION.md evaporates.
    """
    n = len(load_artifacts().catalog)
    quality = load_latent("latent_quality", expected_rows=n)
    assert np.all(quality > 0), "log() is taken of this; it must be positive"
    assert quality.max() / quality.min() > 2, "quality carries no spread"
