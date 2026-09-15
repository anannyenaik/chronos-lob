"""Canonical FI-2010 label and probability-column mapping helpers.

The normalised FI-2010 labels use the raw convention:

* ``1`` = up
* ``2`` = stationary
* ``3`` = down

ChronosLOB prediction artefacts expose named probabilities such as
``prob_down``, ``prob_stationary`` and ``prob_up``.  This module keeps the raw
label convention and the named probability convention tied together explicitly
so metrics and plots never depend on incidental positional ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FI2010_CANONICAL_CLASS_ORDER",
    "FI2010_CLASS_TO_RAW_LABEL",
    "FI2010_PROBABILITY_COLUMN_ORDER",
    "FI2010_RAW_LABEL_TO_CLASS",
    "LabelMappingValidation",
    "canonical_class_name",
    "class_name_to_raw_label",
    "classwise_f1_from_row",
    "labels_to_canonical_class_names",
    "labels_to_raw_labels",
    "probability_columns_for_order",
    "validate_class_order",
    "validate_classwise_f1_columns",
    "validate_confusion_matrix_axis_labels",
    "validate_probability_columns",
]

FI2010_RAW_LABEL_TO_CLASS: dict[int, str] = {
    1: "up",
    2: "stationary",
    3: "down",
}
FI2010_CLASS_TO_RAW_LABEL: dict[str, int] = {
    value: key for key, value in FI2010_RAW_LABEL_TO_CLASS.items()
}
FI2010_CANONICAL_CLASS_ORDER: tuple[str, ...] = ("up", "stationary", "down")
FI2010_PROBABILITY_COLUMN_ORDER: tuple[str, ...] = (
    "prob_up",
    "prob_stationary",
    "prob_down",
)

_CLASS_ALIASES: dict[str, str] = {
    "1": "up",
    "1.0": "up",
    "up": "up",
    "bullish": "up",
    "2": "stationary",
    "2.0": "stationary",
    "stationary": "stationary",
    "unchanged": "stationary",
    "flat": "stationary",
    "neutral": "stationary",
    "3": "down",
    "3.0": "down",
    "down": "down",
    "bearish": "down",
}

_PROBABILITY_ALIASES: dict[str, tuple[str, ...]] = {
    "up": ("prob_up", "probability_up", "probability_1", "probability_1.0"),
    "stationary": (
        "prob_stationary",
        "probability_stationary",
        "probability_2",
        "probability_2.0",
    ),
    "down": ("prob_down", "probability_down", "probability_3", "probability_3.0"),
}


@dataclass(frozen=True)
class LabelMappingValidation:
    """Structured validation result for label-mapping checks."""

    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] | None = None


def canonical_class_name(value: Any) -> str:
    """Return the canonical FI-2010 class name for a raw label or alias."""
    if value is None:
        raise ValueError("FI-2010 label is missing")
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid FI-2010 labels")
    if isinstance(value, int):
        try:
            return FI2010_RAW_LABEL_TO_CLASS[value]
        except KeyError as exc:
            raise ValueError(f"unknown FI-2010 raw label: {value}") from exc
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"unknown FI-2010 raw label: {value}")
        return canonical_class_name(int(value))
    text = str(value).strip().lower()
    if text in _CLASS_ALIASES:
        return _CLASS_ALIASES[text]
    raise ValueError(f"unknown FI-2010 label: {value!r}")


def class_name_to_raw_label(value: str) -> int:
    """Return the FI-2010 raw label for a canonical class name."""
    class_name = canonical_class_name(value)
    return FI2010_CLASS_TO_RAW_LABEL[class_name]


def labels_to_canonical_class_names(values: Sequence[Any]) -> list[str]:
    """Convert raw labels or aliases to canonical class names."""
    return [canonical_class_name(value) for value in values]


def labels_to_raw_labels(values: Sequence[Any]) -> list[int]:
    """Convert canonical names, aliases or raw labels to raw FI-2010 labels."""
    return [class_name_to_raw_label(canonical_class_name(value)) for value in values]


def probability_columns_for_order(
    columns: Sequence[str],
    *,
    class_order: Sequence[str] = FI2010_CANONICAL_CLASS_ORDER,
) -> tuple[str, ...]:
    """Return probability columns aligned to the requested class order.

    The returned tuple is explicit and named, e.g. ``("prob_up",
    "prob_stationary", "prob_down")`` for the canonical order.  Positional
    columns such as ``prob_0``/``prob_1``/``prob_2`` are intentionally refused
    because they are ambiguous against FI-2010's raw label ids.
    """
    validation = validate_probability_columns(columns, class_order=class_order)
    if not validation.passed:
        raise ValueError("; ".join(validation.errors))
    details = validation.details or {}
    order = details.get("probability_column_order_used")
    if not isinstance(order, tuple):
        raise ValueError("probability-column validation did not return an order")
    return order


def validate_probability_columns(
    columns: Sequence[str],
    *,
    class_order: Sequence[str] = FI2010_CANONICAL_CLASS_ORDER,
) -> LabelMappingValidation:
    """Validate and order FI-2010 probability columns by class name."""
    available = {str(column) for column in columns}
    errors: list[str] = []
    warnings: list[str] = []
    ordered: list[str] = []

    ambiguous_positionals = sorted(
        column
        for column in available
        if column.lower() in {"prob_0", "prob_1", "prob_2", "p0", "p1", "p2"}
    )
    if ambiguous_positionals:
        errors.append(
            "ambiguous positional probability columns found: "
            + ", ".join(ambiguous_positionals)
        )

    for raw_class in class_order:
        try:
            class_name = canonical_class_name(raw_class)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        aliases = _PROBABILITY_ALIASES[class_name]
        matches = [alias for alias in aliases if alias in available]
        if not matches:
            errors.append(
                f"missing probability column for FI-2010 class {class_name!r}; "
                f"accepted aliases: {', '.join(aliases)}"
            )
            continue
        if len(matches) > 1:
            preferred = f"prob_{class_name}"
            if preferred in matches:
                warnings.append(
                    f"multiple probability columns map to {class_name!r}; "
                    f"using {preferred!r}"
                )
                ordered.append(preferred)
            else:
                errors.append(
                    f"ambiguous probability columns for FI-2010 class "
                    f"{class_name!r}: {', '.join(matches)}"
                )
        else:
            ordered.append(matches[0])

    canonical_order: tuple[str, ...]
    try:
        canonical_order = tuple(canonical_class_name(item) for item in class_order)
    except ValueError:
        canonical_order = ()
    return LabelMappingValidation(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        details={
            "probability_columns_found": sorted(
                column
                for column in available
                if column.startswith("prob") or column.startswith("probability_")
            ),
            "probability_column_order_used": tuple(ordered),
            "class_order": canonical_order,
        },
    )


def validate_class_order(
    class_order: Sequence[Any],
) -> LabelMappingValidation:
    """Validate the class order used for metrics or probability matrices."""
    try:
        canonical = tuple(canonical_class_name(value) for value in class_order)
    except ValueError as exc:
        return LabelMappingValidation(passed=False, errors=(str(exc),))
    if canonical != FI2010_CANONICAL_CLASS_ORDER:
        return LabelMappingValidation(
            passed=False,
            errors=(
                "FI-2010 class order must be "
                f"{list(FI2010_CANONICAL_CLASS_ORDER)}, got {list(canonical)}",
            ),
            details={"class_order": canonical},
        )
    return LabelMappingValidation(
        passed=True,
        details={"class_order": canonical},
    )


def validate_confusion_matrix_axis_labels(
    labels: Sequence[Any],
) -> LabelMappingValidation:
    """Validate confusion-matrix axes against the canonical FI-2010 order."""
    validation = validate_class_order(labels)
    if not validation.passed:
        return LabelMappingValidation(
            passed=False,
            errors=tuple(
                "confusion matrix axis labels invalid: " + error
                for error in validation.errors
            ),
            details=validation.details,
        )
    return validation


def validate_classwise_f1_columns(columns: Sequence[str]) -> LabelMappingValidation:
    """Validate that class-wise F1 columns are named, not positional."""
    available = {str(column) for column in columns}
    expected = tuple(f"class_f1_{label}" for label in FI2010_CANONICAL_CLASS_ORDER)
    missing = [column for column in expected if column not in available]
    positional = sorted(
        column
        for column in available
        if column.lower().startswith("class_f1_")
        and column.lower() not in set(expected)
    )
    errors: list[str] = []
    warnings: list[str] = []
    if missing:
        errors.append("missing class-wise F1 columns: " + ", ".join(missing))
    if positional:
        warnings.append(
            "non-canonical class-wise F1 columns ignored: " + ", ".join(positional)
        )
    return LabelMappingValidation(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        details={
            "expected_classwise_f1_columns": expected,
            "found_classwise_f1_columns": tuple(
                column for column in expected if column in available
            ),
        },
    )


def classwise_f1_from_row(row: Mapping[str, Any]) -> dict[str, float | None]:
    """Extract named class-wise F1 values in canonical FI-2010 order."""
    validation = validate_classwise_f1_columns(tuple(str(key) for key in row))
    if not validation.passed:
        raise ValueError("; ".join(validation.errors))
    values: dict[str, float | None] = {}
    for class_name in FI2010_CANONICAL_CLASS_ORDER:
        raw = row.get(f"class_f1_{class_name}")
        if raw is None or raw == "":
            values[class_name] = None
            continue
        try:
            values[class_name] = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"class_f1_{class_name} must be numeric or empty"
            ) from exc
    return values
