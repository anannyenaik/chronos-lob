"""Deterministic Binance-style replay helpers.

Helpers in this module load locally captured Binance-style fixtures
(a JSON snapshot and a JSONL stream of diff events) and replay them
through :func:`chronoslob.book.reconstruction.reconstruct_order_book`.
No network calls are made. The functions return in-memory results and
write nothing to disk.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob.book.reconstruction import (
    ReconstructionResult,
    reconstruct_order_book,
)
from chronoslob.data.binance import (
    load_binance_diff_events_jsonl,
    load_binance_snapshot_json,
)

__all__ = [
    "ReplayConfig",
    "replay_binance_jsonl",
    "summarise_replay_result",
]


class ReplayConfig(BaseModel):
    """Configuration for :func:`replay_binance_jsonl`."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    snapshot_path: Path = Field(...)
    updates_path: Path = Field(...)
    symbol: str | None = None
    max_depth: int | None = None
    stop_on_gap: bool = True
    allow_crossed: bool = False

    @field_validator("snapshot_path", "updates_path", mode="before")
    @classmethod
    def _validate_paths(cls, value: object) -> Path:
        if value is None:
            raise ValueError("path must be non-empty")
        if isinstance(value, bool):
            raise TypeError("path must be a string or Path")
        if not isinstance(value, (str, Path)):
            raise TypeError("path must be a string or Path")
        if str(value).strip() == "":
            raise ValueError("path must be non-empty")
        return Path(value)

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("symbol must be a non-empty string when provided")
        return value.strip()

    @field_validator("max_depth")
    @classmethod
    def _validate_max_depth(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("max_depth must be a positive int or None")
        if value <= 0:
            raise ValueError(f"max_depth must be positive; got {value!r}")
        return value


def replay_binance_jsonl(config: ReplayConfig) -> ReconstructionResult:
    """Replay a Binance-style snapshot and diff JSONL file deterministically."""
    snapshot = load_binance_snapshot_json(config.snapshot_path, symbol=config.symbol)
    events = load_binance_diff_events_jsonl(config.updates_path)
    return reconstruct_order_book(
        snapshot,
        events,
        max_depth=config.max_depth,
        stop_on_gap=config.stop_on_gap,
        allow_crossed=config.allow_crossed,
    )


def summarise_replay_result(
    result: ReconstructionResult,
) -> dict[str, str | int | bool | None]:
    """Return a small summary dictionary describing a replay result."""
    return {
        "ok": result.ok,
        "n_snapshots": result.n_snapshots,
        "final_update_id": result.final_update_id,
        "gap_count": result.gap_count,
        "crossed_count": result.crossed_count,
        "issue_count": len(result.issues),
    }
