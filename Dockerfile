# ---------------------------------------------------------------------------
# ReelRank - YouTube-style recommender
#
# The image contains NO deep-learning framework. Every model is either NumPy /
# scikit-learn or hand-written (ALS, co-visitation, MMR, the ANN search), and
# the text encoder is fitted offline and served as a plain matrix. Result: a
# ~450MB image instead of the ~2.5GB a torch-based build would need, which
# matters on a free Hugging Face Space.
#
# Artifacts are BUILT AT IMAGE BUILD TIME (deterministic from config.yaml +
# seed), so the container starts serving immediately instead of training on
# first request.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Many small BLAS ops (96x96 ALS solves); thread contention makes them ~70x
    # slower than single-threaded. Measured, see src/recsys/recall/cf.py.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=7860

WORKDIR /app

# Dependencies first so code edits do not bust the pip layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# HF Spaces runs as a non-root user; these must be writable.
RUN mkdir -p data/raw data/processed artifacts && chmod -R 777 data artifacts

# Build the models into the image. --skip-eval keeps the build short; run
# scripts/05_evaluate.py locally to reproduce the metrics in the docs.
RUN python scripts/build_all.py --skip-eval

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/api/health').status==200 else 1)"

CMD ["python", "-m", "uvicorn", "recsys.api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "7860"]
