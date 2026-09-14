# recon

The reconstruction engine (backend lane) — board detection, per-frame camera pose, VGGT
reconstruction (MASt3R and COLMAP behind a config flag), TSDF fusion, provenance, and the
FastAPI endpoint that `capture/` talks to.

Nothing runs yet. This directory is registered as the `recon` Python package: from the
repository root, `pip install -e .` installs it in editable mode. The first runnable piece
(the reconstruction model on the golden capture) lands with R-01.
