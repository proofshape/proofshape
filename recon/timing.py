"""R-13: a backend-agnostic per-stage timing table, shared by every reconstruction runner.

Every runner already measures its own stages into a `timings: dict[str, float]` (stage name ->
seconds, ending in "total_s") -- this module only renders that dict as a table, so VGGT, MASt3R
and COLMAP all produce the same shape of output without duplicating the formatting three times.
"""

from __future__ import annotations


def format_timing_table(timings: dict[str, float]) -> str:
    """Render `timings` as a table: stage, seconds, share of total.

    `timings` must include a "total_s" key -- every runner's `run_on_capture` already adds one
    as its last stage, and share-of-total is undefined without it.
    """
    if "total_s" not in timings:
        raise ValueError(
            'timings must include a "total_s" key to compute each stage\'s share of it.'
        )

    total_s = timings["total_s"]
    name_width = max(len(name) for name in timings)
    header = f"{'stage':<{name_width}}  {'seconds':>8}  {'share':>7}"
    lines = [header, "-" * len(header)]
    for name, seconds in timings.items():
        share_pct = (seconds / total_s * 100.0) if total_s > 0 else 0.0
        lines.append(f"{name:<{name_width}}  {seconds:>8.2f}  {share_pct:>6.1f}%")
    return "\n".join(lines)
