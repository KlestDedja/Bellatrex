# syntax=docker/dockerfile:1.7
# bellatrex test-runner image
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Build deps for native wheels (numpy/scipy/pandas/etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
  && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Common test tooling preinstalled for speed
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install pytest pytest-cov coverage wheel build

# Helper entrypoint for repeatable test runs
COPY docker/btrex-ci /usr/local/bin/btrex-ci
RUN chmod +x /usr/local/bin/btrex-ci

WORKDIR /work
CMD ["btrex-ci", "--help"]