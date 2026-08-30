# Deployment

**Live:** <https://reelrank-one.vercel.app> — one Python serverless function serving
both the JSON API and the static UI. Docker and Hugging Face Spaces are supported
alternatives and are documented below.

FastAPI serves the API and the UI from the same process, and the models are **built
ahead of time** rather than trained on first request, so the app starts serving
immediately. There is no PyTorch, no `faiss`, no `implicit` — every model is NumPy or
hand-written ([D3](DESIGN_DECISIONS.md)).

---

## Two environments, two dependency sets

This is the single most important thing to know before running anything, because the
two requirement files are not interchangeable:

| | file | contains | used by |
|---|---|---|---|
| **Build / training** | `requirements-build.txt` | pandas, scikit-learn, SciPy, PyArrow, NumPy, FastAPI | `scripts/build_all.py`, every `scripts/*.py`, the test suite |
| **Serving / inference** | `requirements.txt` | NumPy, FastAPI, Uvicorn, Pydantic, PyYAML | the deployed app |

`requirements-build.txt` includes `requirements.txt`, so the build set is a superset.

**Running the pipeline needs the build set.** `pip install -r requirements.txt` followed
by `python scripts/build_all.py` fails at stage 02 — ten of the scripts import pandas or
scikit-learn.

### Why serving has its own requirements file

Training a gradient-boosted model is hard; evaluating one is not. A fitted
`HistGradientBoostingClassifier` is, at prediction time, a list of binary trees plus a
baseline. The histogram binning, the gradient/hessian machinery and the loss objects
exist only to *fit* it. Likewise, a fitted `TfidfVectorizer → TruncatedSVD` pipeline is,
at query time, a vocabulary lookup and a matrix multiply.

So the last build stage ([`scripts/12_export_serving.py`](../scripts/12_export_serving.py))
converts both into plain arrays, and the deployed app loads those instead:

```
artifacts/serving_models.npz    tree arrays, idf vector, SVD components   15.1 MB
artifacts/serving_models.json   vocabulary, stop words, objective weights  0.3 MB
```

Four things this buys, in order of how much they mattered:

1. **It removes a fragile pickle.** A pickled `HistGradientBoostingClassifier` reaches
   for a Cython class whose `__module__` is the bare string `_loss`, so unpickling needs
   a top-level `import _loss` to resolve. That works locally by accident of import order.
   In production it failed outright — see [Vercel, specifically](#vercel-specifically).
2. **Smaller footprint.** scikit-learn + SciPy + joblib is ~200 MB of Linux wheels;
   pandas is another ~55 MB. Removing all four took the deployed bundle from 339 MB
   (over the serverless limit) to comfortably under it.
3. **More portable inference.** The serving path is NumPy and the standard library. No
   version-matched unpickling, so artifacts built today load on a future runtime.
4. **Faster cold start.** Less to import.

**The conversion is verified, not assumed.** The exporter re-scores both paths and
refuses to write a bundle that disagrees by more than `1e-6`:

| model | max \|sklearn − numpy\| | checked on |
|---|---|---|
| ranker (101 trees) | 5.551e-17 | 4,000 feature rows, 5% NaN to exercise missing-value routing |
| head `click` | 5.551e-17 | as above |
| head `long_watch` | 1.110e-16 | as above |
| head `completion` | 5.551e-17 | as above |
| head `liked` | 2.776e-17 | as above |
| head `satisfied` | 1.110e-16 | as above |
| head `dismissed` | 4.337e-19 | as above |
| text encoder | 1.192e-07 | all 6,000 catalog documents |
| query vectors | 1.490e-08 | sample queries, top-3 rankings identical |

The trees agree to machine precision because tree inference is exact arithmetic on the
same thresholds. The encoder's 1.2e-07 is float32 rounding in the SVD component matrix,
which is stored as float32 deliberately ([D3](DESIGN_DECISIONS.md)) — four orders of
magnitude below the gap between adjacent candidates, so it cannot reorder a feed.

[`tests/test_serving_deps.py`](../tests/test_serving_deps.py) enforces the split: it
blocks pandas, scikit-learn, SciPy, joblib and PyArrow at the import hook, then runs a
real recommendation and a real search.

---

## Alternative: Hugging Face Spaces (free, no credit card, no cold-start penalty)

Free Spaces sleep after ~48 h of inactivity and wake in a few seconds — unlike Render's free
tier, which sleeps after 15 minutes and takes ~50 s to wake, long enough that an evaluator's
first click looks broken.

### 1 · Create the Space

Go to <https://huggingface.co/new-space>:

- **Space name**: `reelrank` (or anything)
- **License**: MIT
- **SDK**: **Docker** → *Blank*
- **Hardware**: CPU basic (free)
- **Visibility**: Public

### 2 · Push the code

```bash
git clone https://huggingface.co/spaces/<your-username>/reelrank hf-space
```

Copy the project in, then add the Space card that Hugging Face requires at the repo root:

```bash
cp deploy/huggingface/README.md hf-space/README.md
```

That file carries the YAML front-matter Spaces reads (`sdk: docker`, `app_port: 7860`). It is
kept separate from the project README so GitHub does not render the front-matter as a stray
table.

```bash
cd hf-space && git add -A && git commit -m "Deploy ReelRank" && git push
```

The first build takes ~10 minutes: installing dependencies, then `scripts/build_all.py
--skip-eval` generates the catalog, simulates the watch log, and trains every model inside the
image.

### 3 · Optional — real YouTube data

In the Space UI: **Settings → Variables and secrets → New secret**

| Name | Value |
|---|---|
| `YOUTUBE_API_KEY` | your [Data API v3](https://developers.google.com/youtube/v3/getting-started) key |

Then change the Dockerfile's build line to `--source youtube_api` and push. The catalog becomes
real YouTube videos; nothing else changes, because every source normalises to the same schema.

---

## Local Docker

```bash
docker build -t reelrank .
```

```bash
docker run -p 7860:7860 reelrank
```

Open <http://localhost:7860>. Add `-e YOUTUBE_API_KEY=...` at build time to use real data.

---

## Local, no Docker

Building needs `requirements-build.txt` (pandas, scikit-learn, SciPy). Plain
`requirements.txt` is the **serving** set and deliberately excludes them -- see
[the NumPy-only serving section](#why-serving-has-its-own-requirements-file).

```bash
pip install -r requirements-build.txt && python scripts/build_all.py
```

```bash
python -m uvicorn recsys.api.app:app --app-dir src --port 7860
```

---

## Other hosts

| Host | Notes |
|---|---|
| **Render** | Free web service, git-push deploy. Sleeps after 15 min idle, ~50 s cold start. Set `PORT` — the app reads it. |
| **Railway / Fly.io** | Both take the Dockerfile unchanged. Fly needs `fly launch --dockerfile Dockerfile`. |
| **Google Cloud Run** | Good fit: `gcloud run deploy --source .`, scales to zero, generous free tier. Set `--memory 512Mi` (artifacts ~28 MB; serving is NumPy-only, so peak RSS is ~200 MB). |
| **Vercel** | Where the live demo runs — the whole app, API included, as one serverless function. Needs the NumPy-only serving bundle to fit the size limit, and a rewrite that preserves the path. See [Vercel, specifically](#vercel-specifically). |

Any host that runs a container and gives you a port will work. The app reads `PORT` from the
environment and falls back to `serving.port` in `config.yaml`.

---

## Tuning a running container without rebuilding

Every config value can be overridden by environment variable, using `RECSYS__` and `__` as the
nesting separator:

```bash
docker run -p 7860:7860 -e RECSYS__POLICY__MMR_LAMBDA=0.5 -e RECSYS__SERVING__DEFAULT_N=30 reelrank
```

---

## Production notes

**Health check.** `GET /api/health` returns `{"status": "ok", "ready": true, ...}`. If artifacts
fail to load the app still starts and reports the error there, so a misconfigured deploy shows a
diagnosable page instead of a dead port. The Dockerfile wires this to Docker's `HEALTHCHECK`.

**Threading.** The image pins BLAS to one thread (`OMP_NUM_THREADS=1` etc.). This is not
conservatism — serving does many *small* linear-algebra operations (a 96×96 ALS fold-in per
request), and thread contention makes a single solve ~70× slower
([`recall/cf.py`](../src/recsys/recall/cf.py)). Scale with more *processes*, not more threads:

```bash
uvicorn recsys.api.app:app --app-dir src --workers 4
```

Each worker loads its own ~28 MB of artifacts. Budget ~200 MB RSS per worker.

**Statelessness.** No sessions, no database. Any number of replicas behind a load balancer works
with no shared state.

**Memory.** Artifacts total ~28 MB (serving models 15 MB, item vectors 6 MB, ALS factors
4 MB, catalog 2.5 MB). Serving imports only NumPy, so peak RSS is ~150–200 MB — the
Docker image still carries scikit-learn because it *builds* the artifacts at image build
time, but the running server never imports it.

**Evaluation in the image.** The Docker build runs `--skip-eval` to keep build time down. Run
`python scripts/05_evaluate.py` locally to reproduce every number in
[EVALUATION.md](EVALUATION.md).

---

## Vercel, specifically

The live demo is a single Python function serving both the API and the UI. Two
things about it are non-obvious enough to be worth writing down, because both
produced deployments that built cleanly and then failed at request time.

### 1. The bundle must not need scikit-learn

Vercel's Python runtime prunes files when a bundle exceeds the size limit, and
it decides what is unused by static analysis. Dynamic imports are invisible to
that. A pickled `HistGradientBoostingClassifier` reaches for a Cython class
whose `__module__` is the bare name `_loss` while unpickling, so the pruned
bundle failed with:

```
ModuleNotFoundError: No module named '_loss'
```

Pinning scikit-learn does not fix this — the version was never the cause. The
fix is `scripts/12_export_serving.py`, which converts the trees and the query
encoder into plain arrays so serving needs NumPy and nothing else. That takes
the bundle from 339 MB (over, and therefore pruned) to comfortably under, and
removes the fragile reference at the same time. See
[`src/recsys/serving/trees.py`](../src/recsys/serving/trees.py).

### 2. The rewrite must carry the path

The obvious catch-all rewrite silently breaks routing:

```json
{ "source": "/(.*)", "destination": "/api/index" }
```

Every request then arrives at the function with `scope["path"] == "/api/index"`,
whatever the client asked for, and FastAPI 404s all of it — including its own
`/docs` and `/openapi.json`, which is the giveaway that the route table is fine
and the path is not. The original path is not recoverable from the request
either: dumping the full ASGI scope and every header from the deployed function
showed it absent from both. Only the query string survives.

So the destination has to carry it:

```json
{ "source": "/(.*)", "destination": "/api/index/$1" }
```

and [`api/index.py`](../api/index.py) strips the `/api/index` prefix back off
before Starlette's router sees it. The strip is idempotent, so Docker and the
local dev server — where no rewrite happens — are unaffected.

### Deploying

```bash
vercel --prod
```

`vercel.json` pins 1024 MB and a 30 s ceiling. Cold start is a few seconds
(loading ~28 MB of artifacts); warm requests are the same ~170 ms as local.
