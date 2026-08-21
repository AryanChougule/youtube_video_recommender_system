"""Synthetic YouTube-like catalog generator.

This is the default data source so that the repository is runnable with zero
credentials.  It is *not* a random-noise generator -- it is a deliberate
statistical model of a video catalog, built to reproduce the properties that
actually make recommendation hard:

  1. **Heavy-tailed popularity.**  View counts are log-normal over ~5 orders of
     magnitude, so a naive popularity baseline is genuinely competitive and
     popularity bias is a real problem we have to defeat.
  2. **Latent topical structure.**  Every video has a mixture over the 40
     micro-topics from :mod:`recsys.data.topics`; categories overlap, so
     cross-category affinities exist and are discoverable only from co-watch
     behaviour.
  3. **Channel structure.**  Videos cluster by channel, and channels have their
     own topic drift, so "channel affinity" is a real, separable signal.
  4. **Hidden quality.**  Each video has a latent appeal that is only
     *partially* observable through engagement rate -- so the ranker has
     something real to learn, but cannot learn it perfectly.
  5. **Age dynamics.**  Views accumulate sub-linearly with age, which decouples
     "most viewed" from "trending right now".

The latent topic matrix and quality vector are returned separately: they are
ground truth for evaluation and must never leak into the served catalog.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .topics import (
    CATEGORIES,
    N_TOPICS,
    TOPIC_INDEX,
    TOPIC_SLUGS,
    TOPIC_VOCAB,
    category_topic_weights,
)
from ..clock import reference_now
from .schema import coerce_catalog, join_tags

# --------------------------------------------------------------------------
# Per-topic priors: (relative popularity, median duration in minutes)
# Hand-set to mirror real YouTube patterns -- gaming/entertainment are huge and
# short-form, education/DIY are smaller and long-form.
# --------------------------------------------------------------------------
TOPIC_PRIORS: dict[str, tuple[float, float]] = {
    "pc_hardware": (1.5, 14.0),      "speedrunning": (1.1, 22.0),
    "esports": (1.8, 16.0),          "survival_games": (1.6, 25.0),
    "rpg_lore": (1.3, 20.0),         "mobile_gaming": (1.7, 11.0),
    "web_dev": (1.0, 18.0),          "ml_ai": (1.2, 21.0),
    "devops": (0.7, 24.0),           "gadget_review": (1.9, 13.0),
    "smartphone": (2.2, 12.0),       "productivity": (1.1, 12.0),
    "home_cooking": (1.8, 9.0),      "baking": (1.3, 14.0),
    "street_food": (2.0, 15.0),      "fine_dining": (0.8, 17.0),
    "strength": (1.6, 13.0),         "running": (1.0, 15.0),
    "mobility": (1.2, 20.0),         "nutrition": (1.4, 12.0),
    "personal_finance": (1.5, 14.0), "investing": (1.2, 16.0),
    "startups": (0.8, 19.0),         "space_physics": (1.4, 18.0),
    "biology_health": (1.1, 17.0),   "history": (1.2, 26.0),
    "math_puzzles": (0.9, 15.0),     "music_production": (0.9, 16.0),
    "live_music": (2.1, 8.0),        "music_theory": (0.6, 14.0),
    "film_analysis": (1.3, 22.0),    "comedy": (2.4, 7.0),
    "commentary": (2.3, 19.0),       "travel_budget": (1.2, 16.0),
    "travel_culture": (1.0, 21.0),   "woodwork": (0.8, 23.0),
    "car_review": (1.5, 17.0),       "ev_tech": (1.1, 18.0),
    "skincare": (1.3, 11.0),         "minimalism": (0.9, 13.0),
}

TITLE_TEMPLATES = [
    "How to Master {a} in {n} Minutes",
    "I Tried {a} for {n} Days -- Here Is What Happened",
    "{a}: Everything You Actually Need to Know",
    "The Truth About {a}",
    "{a} vs {b}: Which One Actually Wins?",
    "Why {a} Is Much Harder Than It Looks",
    "{n} {a} Mistakes You Are Probably Making",
    "Building {a} Completely From Scratch",
    "{a}, Explained Simply",
    "My Complete {a} Setup ({year})",
    "A Beginner Guide to {a}",
    "{a} -- 30 Day Results",
    "Stop Getting {a} Wrong",
    "{a}: Ranked From Worst to Best",
    "{a} Tier List ({year})",
    "What Nobody Tells You About {a}",
    "The Full {a} Walkthrough",
    "The Only {a} Guide You Will Ever Need",
    "How {a} Completely Changed My {b}",
    "We Tested {a} So You Do Not Have To",
    "{a} on a Budget: {n} Ideas That Work",
    "I Was Wrong About {a}",
    "{a} in {year}: Is It Still Worth It?",
    "Deep Dive: {a} and {b}",
]

DESCRIPTION_TEMPLATES = [
    "In this video we go deep on {a}. I cover {b}, the setup I use, and the mistakes that cost me months. Timestamps below.",
    "Everything I have learned about {a} in one place -- including {b} and why most guides get it backwards.",
    "A practical, no-fluff walkthrough of {a}. If you have been stuck on {b}, start here.",
    "Part of my ongoing {a} series. This episode focuses on {b} and answers the questions you keep sending me.",
    "I spent months on {a} so you can skip the trial and error. We also touch on {b} near the end.",
]

CHANNEL_PREFIX = ["Pixel", "Iron", "North", "Quiet", "Bright", "Deep", "Open", "Hyper", "Slow", "Bold",
                  "Neon", "Copper", "Atlas", "Vector", "Ember", "Nomad", "Prime", "Lucid", "Grit", "Wren",
                  "Fern", "Halo", "Onyx", "Rift", "Sable", "Terra", "Vault", "Zephyr"]
# Generic suffixes usable anywhere, plus on-brand ones per category. A channel
# called "Dana Kitchen" publishing Gaming videos immediately reads as fake, and
# the UI demo is part of the deliverable -- so naming follows the vertical.
CHANNEL_SUFFIX = ["Forge", "Lab", "Works", "Collective", "Bureau", "Society",
                  "Method", "Desk", "Files", "Notes", "Room", "Depot", "Project"]

CHANNEL_SUFFIX_BY_CATEGORY: dict[str, list[str]] = {
    "Gaming":               ["Gaming", "Plays", "Arena", "Guild", "Quest", "Squad", "Rift"],
    "Science & Technology": ["Labs", "Dev", "Systems", "Bytes", "Stack", "Engineering", "Terminal"],
    "Tech Reviews":         ["Tech", "Reviews", "Gadgets", "Unboxed", "Hardware"],
    "Food":                 ["Kitchen", "Eats", "Table", "Pantry", "Cooks", "Plate", "Appetite"],
    "Health & Fitness":     ["Fitness", "Strength", "Athletics", "Training", "Movement", "Reps"],
    "Finance":              ["Finance", "Capital", "Money", "Wealth", "Markets", "Ledger"],
    "Education":            ["Academy", "Explains", "Institute", "Curious", "Minds", "Archive"],
    "Music":                ["Music", "Sound", "Records", "Sessions", "Audio", "Studio"],
    "Entertainment":        ["Media", "Show", "Comedy", "Reacts", "Screen", "Cut"],
    "Travel":               ["Travels", "Nomad", "Routes", "Wander", "Atlas", "Passport"],
    "Howto & Style":        ["Workshop", "Craft", "Style", "Atelier", "Made", "Home"],
    "Autos & Vehicles":     ["Motors", "Garage", "Auto", "Drives", "Wheels", "Torque"],
    "Sports":               ["Sports", "Athletics", "Field", "League", "Pace", "Bench"],
}
PERSON_NAMES = ["Mira", "Noah", "Priya", "Ravi", "Ines", "Tomas", "Aiko", "Dana", "Kofi", "Lena", "Omar",
                "Sana", "Theo", "Yuki", "Zane", "Ada", "Bruno", "Cleo", "Elias", "Farah", "Gia", "Hugo",
                "Iris", "Jonas", "Kira", "Milo", "Nadia", "Otto", "Rhea", "Sami"]

_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"


def _make_ids(rng: np.random.Generator, n: int, length: int, prefix: str = "") -> list[str]:
    """YouTube-shaped opaque ids, guaranteed unique.

    Collisions are astronomically unlikely at our scale (64**11 keyspace), but
    a silent duplicate would misalign the ground-truth topic matrix against the
    catalog, so we top up explicitly rather than trusting the odds.
    """
    chars = np.array(list(_ID_ALPHABET))
    seen: dict[str, None] = {}
    while len(seen) < n:
        picks = rng.integers(0, len(chars), size=(n - len(seen), length))
        for row in picks:
            seen[prefix + "".join(chars[row])] = None
    return list(seen)


def _dirichlet_around(rng: np.random.Generator, base: np.ndarray, concentration: float) -> np.ndarray:
    """Sample a mixture close to ``base``; higher concentration = closer."""
    alpha = np.maximum(base * concentration, 1e-3)
    return rng.dirichlet(alpha)


def _title_case(term: str) -> str:
    small = {"a", "an", "the", "of", "to", "in", "on", "and", "or", "for", "vs"}
    words = term.split()
    return " ".join(w if (i and w.lower() in small) else (w if w[:1].isupper() else w.capitalize())
                    for i, w in enumerate(words))


def generate_catalog(
    n_videos: int = 6000,
    n_channels: int = 420,
    seed: int = 42,
    reference_date: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Generate a synthetic catalog.

    Returns
    -------
    catalog : DataFrame conforming to ``CATALOG_SCHEMA``
    item_topics : (n_videos, N_TOPICS) row-normalised ground-truth mixtures
    channels : DataFrame of channel metadata (ground truth topic mixtures too)
    """
    rng = np.random.default_rng(seed)
    categories = list(CATEGORIES)

    topic_pop = np.array([TOPIC_PRIORS[s][0] for s in TOPIC_SLUGS])
    topic_dur = np.array([TOPIC_PRIORS[s][1] for s in TOPIC_SLUGS])

    # ---------------- channels ----------------
    # Channels-per-category is itself skewed: some verticals are far more
    # crowded than others, exactly as on the real platform.
    cat_share = rng.dirichlet(np.full(len(categories), 3.0))
    chan_cat_idx = rng.choice(len(categories), size=n_channels, p=cat_share)

    channel_ids = _make_ids(rng, n_channels, 22, prefix="UC")
    channel_topics = np.zeros((n_channels, N_TOPICS))
    channel_names: list[str] = []

    for c in range(n_channels):
        cat = categories[chan_cat_idx[c]]
        base = np.zeros(N_TOPICS)
        for slug, w in category_topic_weights(cat).items():
            base[TOPIC_INDEX[slug]] = w
        # concentration 14 -> channel sits near its category but drifts
        channel_topics[c] = _dirichlet_around(rng, base, 14.0)

        # 75% on-brand suffix for the vertical, 25% generic.
        pool = CHANNEL_SUFFIX_BY_CATEGORY.get(cat, CHANNEL_SUFFIX)
        suffix = str(rng.choice(pool if rng.random() < 0.75 else CHANNEL_SUFFIX))
        style = rng.integers(0, 4)
        if style == 0:
            name = f"{rng.choice(CHANNEL_PREFIX)}{suffix}"
        elif style == 1:
            name = f"The {rng.choice(CHANNEL_PREFIX)} {suffix}"
        elif style == 2:
            name = f"{rng.choice(PERSON_NAMES)} {suffix}"
        else:
            name = f"{rng.choice(PERSON_NAMES)}{suffix}"
        channel_names.append(name)

    # Channel reach is heavy-tailed: a few mega-channels, a long tail of small ones.
    channel_scale = rng.lognormal(mean=0.0, sigma=1.15, size=n_channels)

    # Videos per channel follows the same skew (big channels publish more).
    pub_weight = channel_scale ** 0.6
    pub_weight = pub_weight / pub_weight.sum()
    video_channel = rng.choice(n_channels, size=n_videos, p=pub_weight)

    # ---------------- videos ----------------
    item_topics = np.zeros((n_videos, N_TOPICS))
    for v in range(n_videos):
        # concentration 26 -> a video is tightly focused within its channel
        item_topics[v] = _dirichlet_around(rng, channel_topics[video_channel[v]], 26.0)

    dominant = item_topics.argmax(axis=1)
    # secondary topic drives the "{b}" slot, creating genuine bridge titles
    secondary = np.argsort(-item_topics, axis=1)[:, 1]

    # Latent appeal -- only partially observable via engagement rate.
    quality = rng.lognormal(mean=0.0, sigma=0.55, size=n_videos)

    # Publication dates: skewed towards recent (a live catalog keeps growing).
    max_age = 1000
    age_days = (rng.beta(1.6, 2.6, size=n_videos) * max_age).round()
    now = reference_now(reference_date)
    published_at = now - pd.to_timedelta(age_days, unit="D")
    published_at = published_at + pd.to_timedelta(rng.integers(0, 86400, n_videos), unit="s")

    # Views: log-normal driven by channel reach x topic popularity x quality,
    # accumulated sub-linearly with age so "most viewed" != "trending".
    base_rate = (
        420.0
        * channel_scale[video_channel]
        * topic_pop[dominant]
        * quality
        * rng.lognormal(0.0, 1.25, size=n_videos)
    )
    accumulation = np.power(np.maximum(age_days, 1.0), 0.72)
    view_count = np.maximum(12, (base_rate * accumulation)).astype(np.int64)

    # Engagement rates depend on quality but are noisy -> the ranker can learn
    # a useful-but-imperfect proxy for the hidden appeal.
    like_rate = np.clip(0.028 * quality * rng.lognormal(0.0, 0.38, n_videos), 0.001, 0.22)
    comment_rate = np.clip(0.0022 * quality * rng.lognormal(0.0, 0.55, n_videos), 0.0, 0.05)
    like_count = (view_count * like_rate).astype(np.int64)
    comment_count = (view_count * comment_rate).astype(np.int64)

    # Duration: topic median, log-normal spread, plus ~8% Shorts.
    duration_min = topic_dur[dominant] * rng.lognormal(0.0, 0.45, n_videos)
    is_short = rng.random(n_videos) < 0.08
    duration_min = np.where(is_short, rng.uniform(0.25, 1.0, n_videos), duration_min)
    duration_seconds = np.maximum(15, (duration_min * 60)).astype(np.int64)

    # ---------------- text ----------------
    years = published_at.year.to_numpy()
    titles: list[str] = []
    descriptions: list[str] = []
    tags_col: list[str] = []

    tmpl_idx = rng.integers(0, len(TITLE_TEMPLATES), n_videos)
    desc_idx = rng.integers(0, len(DESCRIPTION_TEMPLATES), n_videos)
    numbers = rng.choice([3, 5, 7, 10, 12, 15, 20, 30, 100], size=n_videos)

    for v in range(n_videos):
        vocab_a = TOPIC_VOCAB[TOPIC_SLUGS[dominant[v]]]
        vocab_b = TOPIC_VOCAB[TOPIC_SLUGS[secondary[v]]]
        term_a = str(rng.choice(vocab_a))
        term_b = str(rng.choice(vocab_b))

        titles.append(
            TITLE_TEMPLATES[tmpl_idx[v]].format(
                a=_title_case(term_a), b=_title_case(term_b),
                n=int(numbers[v]), year=int(years[v]),
            )
        )
        descriptions.append(
            DESCRIPTION_TEMPLATES[desc_idx[v]].format(a=term_a, b=term_b)
        )

        # Tags are drawn from the top-3 latent topics -> tags carry real signal
        # about the mixture, which is what makes content recall work.
        top3 = np.argsort(-item_topics[v])[:3]
        tag_pool: list[str] = []
        for rank, t in enumerate(top3):
            k = 3 if rank == 0 else 2
            pool = TOPIC_VOCAB[TOPIC_SLUGS[t]]
            tag_pool += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        tag_pool.append(TOPIC_SLUGS[dominant[v]].replace("_", " "))
        tags_col.append(join_tags(list(dict.fromkeys(tag_pool))))

    catalog = pd.DataFrame({
        "video_id": _make_ids(rng, n_videos, 11),
        "title": titles,
        "channel_id": [channel_ids[c] for c in video_channel],
        "channel_title": [channel_names[c] for c in video_channel],
        "category": [categories[chan_cat_idx[c]] for c in video_channel],
        "tags": tags_col,
        "description": descriptions,
        "published_at": published_at,
        "duration_seconds": duration_seconds,
        "view_count": view_count,
        "like_count": like_count,
        "comment_count": comment_count,
        "thumbnail_url": "",          # UI renders a deterministic gradient
        "source": "synthetic",
    })
    catalog = coerce_catalog(catalog)
    # Row order and count must survive coercion, otherwise ``item_topics`` and
    # ``quality`` would silently misalign against the catalog. Ids are unique
    # and titles are always non-empty by construction, so this must hold.
    if len(catalog) != n_videos:
        raise RuntimeError(
            f"catalog coercion dropped rows ({n_videos} -> {len(catalog)}); "
            "ground-truth arrays would misalign"
        )
    catalog["latent_quality"] = quality      # stripped before serving

    channels = pd.DataFrame({
        "channel_id": channel_ids,
        "channel_title": channel_names,
        "category": [categories[i] for i in chan_cat_idx],
        "reach": channel_scale,
    })

    item_topics = item_topics / item_topics.sum(axis=1, keepdims=True)
    return catalog, item_topics, channels
