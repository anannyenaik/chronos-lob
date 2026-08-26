"""Canonical public wording shared by the report and evidence-pack builders.

The final empirical report, the evidence pack and the SSL-v2 analysis each
render the same scope and provenance statements. Keeping one copy here means a
scope change is edited once and cannot drift between the generated documents,
which the cross-document consistency tests would otherwise have to police.

These strings are claim boundaries. Widening one widens every public report
that renders it, so change them only alongside the retained evidence.
"""

from __future__ import annotations

__all__ = [
    "BROADER_PROPER_TRAINING_PARAGRAPH",
    "HAMILTON_PROVENANCE_PARAGRAPH",
    "NEURAL_LIMITATION_PARAGRAPH",
    "SSL_V2_SCOPE_PARAGRAPH",
]

SSL_V2_SCOPE_PARAGRAPH = (
    "The SSL-v2 benchmark is complete for the stored FI-2010 scope: folds 1-5, "
    "horizons 10/50, seeds 0-2 and lookback 50. Across 30 matched comparison "
    "cells, SSL-v2 has positive mean deltas for macro-F1, MCC, ECE and Brier, "
    "supporting scoped predictive and calibration improvement for this exact "
    "retained scope. The evidence is mixed by seed and horizon, including negative "
    "mean macro-F1 deltas for seed 1 and horizon 50, so broad SSL improvement "
    "remains unsupported."
)

HAMILTON_PROVENANCE_PARAGRAPH = (
    "The seed-1 and seed-2 SSL-v2 refresh was executed as independent Slurm array "
    "jobs on Durham University Hamilton/NCC HPC. Retained summaries, provenance and "
    "claim assessments are committed; large checkpoints, raw predictions and "
    "cluster logs are intentionally excluded. GPU determinism warnings are "
    "documented, and bitwise reproducibility is not claimed."
)

NEURAL_LIMITATION_PARAGRAPH = (
    "The one-epoch neural full grid is matched comparison evidence, not a "
    "performance-maximising neural benchmark. The proper-training neural subset "
    "remains partial, and a broader proper-training neural benchmark across folds, "
    "seeds, lookbacks and model families is deferred."
)

BROADER_PROPER_TRAINING_PARAGRAPH = (
    "The broader proper-training neural benchmark is now complete as post-v0.2.0 work. "
    "It covers 180 Hamilton Slurm cells across folds 1-5, seeds 0-2, lookbacks "
    "20/50/100, horizons 10/50, and matrix-transformer plus DeepLOB-style model "
    "families. The matrix transformer has stronger mean macro-F1 and MCC than the "
    "DeepLOB-style model in the retained benchmark, but with substantially higher "
    "variability and weak lookback-100 behaviour. Confidence filtering improves "
    "retained-sample metrics while reducing active fraction. The result supports a "
    "scoped benchmark comparison, not a broad neural-superiority claim. v0.2.0 remains "
    "the published release and does not include this post-release benchmark."
)
