# syntax=docker/dockerfile:1.4
#
# Build cmd:
#
# docker build --secret id=internal_ca,src=$CERTS -f docker/gaudi.env.Dockerfile -t gaudi-env-cafl4ds:latest .
#
# CERTS:            <your/PEM/or/CRT/file>. Adding this --secret is optional.

# ---- STAGE 1: Builder ------------------------------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS builder

# Bring in uv just for the lightning-fast export
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Export requirements text locally (No internet or certs required here!)
RUN uv export --frozen --no-dev --extra gaudi --no-hashes --no-emit-project --format requirements-txt > requirements.txt

# Filter out hardware-specific packages to protect the base image binaries
RUN grep -vE '^(torch|torchvision|torchaudio|vllm|nvidia|triton)' requirements.txt > requirements.final.txt

# ---- STAGE 2: Production ---------------------------------------------------------------------------------------------
# Base Image MUST match driver version (see hl-smi)
# Using the official Habana PyTorch installer image
#
# NOTE: to take advantage of these hardware-optimized binaries, we can't run in a local .venv, hence
# `uv pip install ... --system --break-system-packages` instead of `uv sync...`
FROM vault.habana.ai/gaudi-docker/1.24.0/ubuntu22.04/habanalabs/pytorch-installer-2.10.0:latest

# Certificates
RUN --mount=type=secret,id=internal_ca \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    if [ -f /run/secrets/internal_ca ]; then \
        cp /run/secrets/internal_ca /usr/local/share/ca-certificates/internal-ca.crt && \
        update-ca-certificates; \
    fi && \
    rm -rf /var/lib/apt/lists/*

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    UV_CERT_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /workspace

COPY --link --from=builder /bin/uv /bin/uvx /bin/
COPY --link --from=builder /app/requirements.final.txt .
COPY --link --from=builder /app/pyproject.toml .

# Install dependencies
# 1. We use --no-deps so uv strictly installs ONLY what is in our filtered list
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "$(which python)" --system --break-system-packages --no-deps -r requirements.final.txt

# Set HF_HOME to the mounted directory for HF cache. This ensures we re-download models if absolutely necessary
# PYTHONPATH is set defensively here to take whatever may be set in the base image.
ENV HF_HOME="/workspace/cache" \
    PYTHONPATH="/workspace:${PYTHONPATH}"

# Expect application code to be mounted at runtime.
CMD ["/bin/bash"]
