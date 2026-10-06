#!/usr/bin/env bash
set -euo pipefail

# R-15 container entrypoint. Fetch the VGGT weights before serving, so a failed download stops
# the container loudly here instead of failing the first user's session with a misleading
# reconstruction error. Weights go to HF_HOME at start-up and are never baked into the image.
if [ "${PROOFSHAPE_PREFETCH_MODEL:-1}" != "0" ]; then
  python - <<'PY'
from huggingface_hub import snapshot_download

from recon.vggt_runner import DEFAULT_MODEL_ID

path = snapshot_download(DEFAULT_MODEL_ID)
print(f"Model {DEFAULT_MODEL_ID} ready at {path}", flush=True)
PY
fi

# Exactly one worker: SessionStore keeps sessions on local disk behind an in-process lock, so a
# second worker would answer session_not_found for sessions the first one created.
# Trust the platform proxy's X-Forwarded-Proto: TLS ends at Lightning's proxy, and without this
# getReconstruction's glb_url (built by request.url_for) would come back as http://.
exec uvicorn recon.service:app --host 0.0.0.0 --port 8000 --workers 1 \
  --proxy-headers --forwarded-allow-ips '*'
