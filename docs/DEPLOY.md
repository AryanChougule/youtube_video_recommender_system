# Deployment

The app is a single Docker container: FastAPI serves both the JSON API and the static UI, and
the models are **built into the image** so the container starts serving immediately rather than
training on first request.

Image size ≈ 450 MB. There is no PyTorch, no `faiss`, no `implicit` — every model is NumPy /
scikit-learn or hand-written, and the text encoder is fitted offline and served as a plain
matrix ([D3](DESIGN_DECISIONS.md)).

---

## Recommended: Hugging Face Spaces (free, no credit card, no cold-start penalty)

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

```bash
pip install -r requirements.txt && python scripts/build_all.py
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
| **Google Cloud Run** | Good fit: `gcloud run deploy --source .`, scales to zero, generous free tier. Set `--memory 1Gi` (artifacts are ~25 MB but pandas/sklearn need headroom). |
| **Vercel** | Frontend only — the Python API needs a container host. Would require splitting the UI out and handling CORS. |

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

Each worker loads its own ~25 MB of artifacts. Budget ~400 MB RSS per worker.

**Statelessness.** No sessions, no database. Any number of replicas behind a load balancer works
with no shared state.

**Memory.** Artifacts total ~25 MB (item vectors 6 MB, text encoder 14 MB, ALS factors 4 MB).
Peak RSS ≈ 350–400 MB with pandas and scikit-learn loaded.

**Evaluation in the image.** The Docker build runs `--skip-eval` to keep build time down. Run
`python scripts/05_evaluate.py` locally to reproduce every number in
[EVALUATION.md](EVALUATION.md).
