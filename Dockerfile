# R-15: the reconstruction service (recon/service.py, R-14) as one GPU image.
# Pinned to the CUDA PyTorch build validated on the shared Lightning Studio (requirements-gpu.txt);
# torch is never reinstalled from PyPI, which could swap in a mismatched build.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.source="https://github.com/proofshape/proofshape" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.description="ProofShape reconstruction service (VGGT backend)"

# git: requirements-gpu.txt pins VGGT to an upstream commit via git+https.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing recon/ doesn't reinstall them. requirements.txt's `-e .` line
# needs the source, so the project itself is installed after it's copied in.
COPY requirements.txt requirements-gpu.txt ./
RUN grep -v '^-e ' requirements.txt > /tmp/requirements-deps.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-deps.txt -r requirements-gpu.txt \
    && python -m pip uninstall -y opencv-python \
    && python -m pip install --no-cache-dir --force-reinstall --no-deps \
        "$(grep '^opencv-python-headless==' requirements.txt)"

COPY pyproject.toml ./
COPY recon/ recon/
COPY scripts/container_entrypoint.sh /usr/local/bin/container_entrypoint.sh
RUN python -m pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 1000 proofshape \
    && mkdir -p /data/sessions /data/hf \
    && chown -R proofshape:proofshape /data

ENV PROOFSHAPE_DATA_DIR=/data/sessions \
    HF_HOME=/data/hf \
    PYTHONUNBUFFERED=1

USER proofshape
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/container_entrypoint.sh"]
