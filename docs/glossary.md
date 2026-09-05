# Glossary

Project vocabulary, so the same word means the same thing in a story, a pull request and a
review. Terms that carry a specific meaning here are marked **(ProofShape term)**.

| Term | Meaning |
|---|---|
| **ChArUco board** | A checkerboard with a unique marker embedded in each square, so every corner is individually identifiable. Printed on A4 or Letter. Supplies scale, a metric camera pose per frame, and the per-order code. |
| **Reference sheet** | Whatever known-size object is in frame: the ChArUco board (tier 1), a blank sheet (tier 2), or a credit card (tier 3). Without one, no dimensional claims are made. |
| **Metric** | In real-world units. A reconstruction is metric once it has been scaled against the reference sheet. |
| **Golden capture** | The committed fixture set: several objects photographed on a printed board, stored outside git behind a download script. The only source of real photographs during the backend push. |
| **Provenance** **(ProofShape term)** | Per-vertex label saying whether the camera actually saw that surface. **Observed** means two or more frames with adequate triangulation baseline. Everything else is **Unobserved**. |
| **σ (sigma)** | Per-vertex uncertainty, computed from the spread of depth residuals across the frames that observe a vertex. Not the spread across generative samples — see D-006. |
| **σ_floor** | The smallest deviation the method can distinguish from its own noise, measured per reference tier in the October calibration study. Every demo tolerance is set from it. |
| **Disagreement map** **(ProofShape term)** | Where the generative completion samples differ from each other. Drives capture guidance only; never enters a verdict. |
| **The hard rule** **(ProofShape term)** | Unobserved geometry can never pass or fail a part. It may only trigger a Rescan. See D-005. |
| **TSDF fusion** | Truncated signed distance function. Merges per-frame depth maps into one volume, from which a mesh is extracted. |
| **Deviation heatmap** | The 3D model coloured by signed distance to the CAD surface. Positive is extra material, negative is missing. Unobserved regions stay grey rather than neutral. |
| **Pass / Fail / Rescan / Unverifiable** | The four verdicts. Unverifiable is declared at intake, before any photo, when a tolerance sits below σ_floor for the required reference tier. |
| **Tier** | How essential a feature is. Tier 1 must work, Tier 2 should, Tier 3 if there is time. |
| **Lane** | A directory in the repository: `/recon`, `/capture`, `/inspect`. A place, not a person — see D-020. |
| **The chain** **(ProofShape term)** | The longest run of dependent stories, about 42 hours. It advances at one person's pace no matter how many people are free, and it is what actually sets the finish date. |
| **Integration point (I1–I4)** | A scheduled day when all other work stops and the three parts are wired together and proven. |
| **Review (R1–R7)** | The biweekly progress presentation to the class. A graded deliverable. |
| **Backbone** | The reconstruction model: VGGT, with MASt3R and COLMAP behind the same config flag. A large learned model, not a generative one — see D-018. |
