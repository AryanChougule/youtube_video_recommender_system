"""Latent micro-topic vocabulary shared by the catalog generator and the
user simulator.

Why micro-topics instead of plain categories
--------------------------------------------
If a user's taste were simply "likes Gaming", the whole recommendation problem
would collapse into a category lookup and there would be nothing for
collaborative filtering to discover.  Real taste is finer-grained and *crosses*
category boundaries: someone who watches "Budget Gaming PC Build" is expressing
an affinity for PC HARDWARE, which lives in both Gaming and Tech.

So we model 40 latent micro-topics.  Categories are overlapping distributions
over them, which creates genuine cross-category "bridges".  Content-based
filtering cannot see those bridges (the words differ); category rules cannot
see them (the labels differ); collaborative filtering *can*, because co-watch
behaviour reveals them.  That gap is precisely what we want the hybrid to fill
-- and because we generated it, we can measure whether it does.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 40 latent micro-topics: (slug, display name, vocabulary)
# --------------------------------------------------------------------------
MICRO_TOPICS: list[tuple[str, str, list[str]]] = [
    ("pc_hardware", "PC Hardware", ["GPU", "RTX 5080", "motherboard", "CPU cooler", "DDR5 RAM", "power supply", "thermal paste", "custom loop", "case airflow", "overclocking"]),
    ("speedrunning", "Speedrunning", ["any percent run", "world record", "frame perfect glitch", "leaderboard", "route optimisation", "tool assisted run", "sequence break", "reset grind"]),
    ("esports", "Esports", ["grand final", "tournament", "patch notes", "the meta", "draft phase", "clutch play", "roster change", "ranked ladder"]),
    ("survival_games", "Survival Games", ["base building", "crafting recipe", "hardcore mode", "world seed", "biome", "day one survival", "resource farm"]),
    ("rpg_lore", "RPG Lore and Builds", ["questline", "endgame build", "boss fight", "hidden lore", "skill tree", "new game plus", "secret ending"]),
    ("mobile_gaming", "Mobile Gaming", ["gacha pulls", "tier list", "banner review", "free to play progress", "reroll guide", "auto battler"]),
    ("web_dev", "Web Development", ["React", "Next.js", "TypeScript", "Tailwind CSS", "REST API", "server components", "state management", "deploy pipeline"]),
    ("ml_ai", "Machine Learning and AI", ["transformer", "large language model", "fine-tuning", "embeddings", "PyTorch", "training dataset", "attention", "inference cost"]),
    ("devops", "DevOps and Homelab", ["Docker", "Kubernetes", "CI/CD", "Linux server", "self-hosted", "homelab rack", "reverse proxy", "backup strategy"]),
    ("gadget_review", "Gadget Reviews", ["unboxing", "hands-on", "teardown", "first impressions", "build quality", "long-term review"]),
    ("smartphone", "Smartphones", ["flagship", "battery life", "display test", "chipset", "camera comparison", "night mode", "charging speed"]),
    ("productivity", "Productivity Systems", ["Notion setup", "note-taking", "second brain", "weekly review", "task automation", "deep work"]),
    ("home_cooking", "Home Cooking", ["weeknight dinner", "one pot meal", "meal prep", "budget recipe", "family dinner", "pantry staples", "30 minute meal"]),
    ("baking", "Baking", ["sourdough starter", "laminated dough", "buttercream", "proofing", "open crumb", "brown butter", "pastry"]),
    ("street_food", "Street Food", ["night market", "hawker stall", "food tour", "local eats", "hole in the wall", "regional specialty"]),
    ("fine_dining", "Fine Dining Technique", ["tasting menu", "plating", "mother sauce", "knife skills", "sous vide", "reduction", "michelin star"]),
    ("strength", "Strength Training", ["hypertrophy", "progressive overload", "compound lift", "push pull legs", "deadlift form", "training split"]),
    ("running", "Running and Endurance", ["5k time", "marathon block", "tempo run", "zone 2", "cadence", "VO2 max", "race pace"]),
    ("mobility", "Yoga and Mobility", ["morning flow", "hip opener", "breathwork", "flexibility routine", "recovery day", "desk posture"]),
    ("nutrition", "Nutrition", ["macros", "protein intake", "cutting phase", "lean bulk", "meal timing", "hydration", "supplements"]),
    ("personal_finance", "Personal Finance", ["index fund", "emergency fund", "zero-based budget", "compound interest", "debt payoff", "credit score"]),
    ("investing", "Investing and Markets", ["valuation", "dividend yield", "portfolio allocation", "earnings call", "market cycle", "risk management"]),
    ("startups", "Startups", ["bootstrapped", "monthly recurring revenue", "product-market fit", "cold outreach", "fundraising", "solo founder"]),
    ("space_physics", "Space and Physics", ["black hole", "general relativity", "quantum mechanics", "telescope image", "orbital mechanics", "dark matter"]),
    ("biology_health", "Biology and Health Science", ["immune system", "gut microbiome", "sleep architecture", "neuroscience", "clinical trial", "metabolism"]),
    ("history", "History", ["ancient empire", "primary source", "dynasty", "revolution", "archive footage", "forgotten war"]),
    ("math_puzzles", "Maths and Puzzles", ["elegant proof", "paradox", "probability puzzle", "geometry", "infinity", "counterintuitive result"]),
    ("music_production", "Music Production", ["mixing", "sample pack", "analog synth", "DAW workflow", "mastering chain", "sound design"]),
    ("live_music", "Live Performance", ["live session", "acoustic set", "tiny desk", "encore", "setlist", "crowd singalong"]),
    ("music_theory", "Music Theory", ["chord progression", "modal interchange", "voicing", "key change", "counterpoint", "time signature"]),
    ("film_analysis", "Film and Video Craft", ["cinematography", "script structure", "editing rhythm", "director style", "single take shot", "colour grade"]),
    ("comedy", "Comedy and Sketch", ["crowd work", "punchline", "improv scene", "sketch", "roast", "deadpan"]),
    ("commentary", "Reactions and Commentary", ["first time watching", "full breakdown", "hot take", "ranking every", "explained", "deep dive"]),
    ("travel_budget", "Budget Travel", ["hostel review", "carry-on only", "long layover", "itinerary", "visa run", "cheap flights"]),
    ("travel_culture", "Culture and Slow Travel", ["local guide", "homestay", "village festival", "regional dialect", "morning market"]),
    ("woodwork", "Woodworking and DIY", ["joinery", "router jig", "workbench build", "hand-cut dovetail", "wood finish", "shop tour"]),
    ("car_review", "Car Reviews", ["test drive", "torque curve", "suspension tuning", "zero to sixty", "trim levels", "daily driver"]),
    ("ev_tech", "EV Technology", ["battery pack", "charging curve", "real range test", "electric motor", "regen braking", "road trip test"]),
    ("skincare", "Skincare and Grooming", ["retinol", "SPF", "evening routine", "skin barrier", "vitamin C serum", "patch test"]),
    ("minimalism", "Minimal Living", ["declutter", "capsule wardrobe", "small space", "tiny home", "one bag", "slow living"]),
]

TOPIC_SLUGS = [t[0] for t in MICRO_TOPICS]
TOPIC_NAMES = {t[0]: t[1] for t in MICRO_TOPICS}
TOPIC_VOCAB = {t[0]: t[2] for t in MICRO_TOPICS}
N_TOPICS = len(MICRO_TOPICS)
TOPIC_INDEX = {slug: i for i, slug in enumerate(TOPIC_SLUGS)}

# --------------------------------------------------------------------------
# Categories = overlapping mixtures over micro-topics.
# The repeated slugs across categories ARE the cross-category bridges: they are
# what makes "Gaming -> PC Hardware -> Tech Reviews" a discoverable path.
# --------------------------------------------------------------------------
CATEGORIES: dict[str, dict[str, float]] = {
    "Gaming":               {"speedrunning": 1.0, "esports": 1.0, "survival_games": 1.0, "rpg_lore": 1.0, "mobile_gaming": 0.8, "pc_hardware": 0.55, "commentary": 0.30},
    "Science & Technology": {"web_dev": 1.0, "ml_ai": 1.0, "devops": 0.9, "pc_hardware": 0.8, "productivity": 0.5, "math_puzzles": 0.25},
    "Tech Reviews":         {"gadget_review": 1.0, "smartphone": 1.0, "pc_hardware": 0.7, "ev_tech": 0.35, "productivity": 0.3},
    "Food":                 {"home_cooking": 1.0, "baking": 1.0, "street_food": 0.9, "fine_dining": 0.8, "nutrition": 0.3},
    "Health & Fitness":     {"strength": 1.0, "running": 1.0, "mobility": 0.9, "nutrition": 0.9, "biology_health": 0.35},
    "Finance":              {"personal_finance": 1.0, "investing": 1.0, "startups": 0.8, "productivity": 0.25},
    "Education":            {"space_physics": 1.0, "biology_health": 1.0, "history": 1.0, "math_puzzles": 1.0, "ml_ai": 0.3},
    "Music":                {"music_production": 1.0, "live_music": 1.0, "music_theory": 0.9, "film_analysis": 0.2},
    "Entertainment":        {"comedy": 1.0, "commentary": 1.0, "film_analysis": 0.9, "live_music": 0.3},
    "Travel":               {"travel_budget": 1.0, "travel_culture": 1.0, "street_food": 0.6, "minimalism": 0.3},
    "Howto & Style":        {"woodwork": 1.0, "skincare": 1.0, "minimalism": 0.9, "productivity": 0.3, "pc_hardware": 0.2},
    "Autos & Vehicles":     {"car_review": 1.0, "ev_tech": 1.0, "pc_hardware": 0.2},
    "Sports":               {"running": 0.9, "strength": 0.7, "esports": 0.6, "commentary": 0.4},
}

CATEGORY_NAMES = list(CATEGORIES)


def category_topic_weights(category: str) -> dict[str, float]:
    """Normalised micro-topic mixture for a category (falls back to uniform)."""
    weights = CATEGORIES.get(category)
    if not weights:
        return {slug: 1.0 / N_TOPICS for slug in TOPIC_SLUGS}
    total = sum(weights.values())
    return {slug: w / total for slug, w in weights.items()}
