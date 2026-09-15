"""Tests for the official FI-2010 ``.txt`` matrix adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from chronoslob.cli import (
    _convert_fi2010_official_impl,
    _verify_fi2010_local_impl,
)
from chronoslob.data.fi2010 import FI2010Config, load_fi2010
from chronoslob.data.fi2010_official import (
    OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT,
    OFFICIAL_FI2010_LABEL_HORIZONS,
    OFFICIAL_FI2010_LEVEL_COUNT,
    OFFICIAL_FI2010_LOB_ROW_COUNT,
    OFFICIAL_FI2010_ROW_COUNT,
    OfficialFI2010ConversionReport,
    OfficialFI2010InspectionReport,
    build_official_fi2010_column_names,
    convert_official_fi2010_to_csv,
    inspect_official_fi2010_file,
)


def _write_synthetic_official_file(
    tmp_path: Path,
    *,
    n_snapshots: int = 6,
    name: str = "Train_Dst_NoAuction_ZScore_CF_1.txt",
) -> Path:
    """Build a tiny synthetic FI-2010-shaped .txt matrix in ``tmp_path``."""
    rows: list[list[float | int | str]] = []
    for level_index in range(OFFICIAL_FI2010_LEVEL_COUNT):
        level = level_index + 1
        ask_price_row: list[float | int | str] = []
        ask_size_row: list[float | int | str] = []
        bid_price_row: list[float | int | str] = []
        bid_size_row: list[float | int | str] = []
        for sample in range(n_snapshots):
            ask_price = 100.0 + 0.10 * level + 0.001 * sample
            bid_price = 100.0 - 0.10 * level - 0.001 * sample
            ask_size_row.append(10 + level + sample)
            bid_size_row.append(11 + level + sample)
            ask_price_row.append(round(ask_price, 5))
            bid_price_row.append(round(bid_price, 5))
        rows.append(ask_price_row)
        rows.append(ask_size_row)
        rows.append(bid_price_row)
        rows.append(bid_size_row)
    for feature_index in range(OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT):
        rows.append(
            [
                round(0.001 * (feature_index + 1) + 0.0001 * sample, 6)
                for sample in range(n_snapshots)
            ],
        )
    label_cycle = ("1", "2", "3")
    for horizon_index in range(len(OFFICIAL_FI2010_LABEL_HORIZONS)):
        rows.append(
            [
                label_cycle[(sample + horizon_index) % len(label_cycle)]
                for sample in range(n_snapshots)
            ],
        )
    assert len(rows) == OFFICIAL_FI2010_ROW_COUNT
    output = tmp_path / name
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(" ".join(str(value) for value in row))
            handle.write("\n")
    return output


# ---------------------------------------------------------------------------
# Column name helper
# ---------------------------------------------------------------------------


def test_build_official_fi2010_column_names_default_layout() -> None:
    columns = build_official_fi2010_column_names()
    expected_total = (
        OFFICIAL_FI2010_LOB_ROW_COUNT
        + OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT
        + len(OFFICIAL_FI2010_LABEL_HORIZONS)
    )
    assert len(columns) == expected_total
    assert columns[0] == "bid_price_1"
    assert columns[1] == "bid_quantity_1"
    assert columns[2] == "ask_price_1"
    assert columns[3] == "ask_quantity_1"
    assert "f_001" in columns
    assert "f_104" in columns
    assert columns[-5:] == [f"label_{h}" for h in OFFICIAL_FI2010_LABEL_HORIZONS]


def test_build_official_fi2010_column_names_with_split_column() -> None:
    columns = build_official_fi2010_column_names(include_split_column=True)
    assert columns[0] == "split"
    assert columns.count("split") == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"level_count": 0},
        {"handcrafted_count": -1},
        {"label_horizons": ()},
        {"label_horizons": (10, 10)},
        {"label_horizons": (-1,)},
    ],
)
def test_build_official_fi2010_column_names_rejects_invalid_args(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        build_official_fi2010_column_names(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def test_inspect_official_fi2010_file_recognises_official_layout(tmp_path: Path) -> None:
    fixture = _write_synthetic_official_file(tmp_path, n_snapshots=5)
    report = inspect_official_fi2010_file(fixture)
    assert isinstance(report, OfficialFI2010InspectionReport)
    assert report.path == fixture
    assert report.row_count == OFFICIAL_FI2010_ROW_COUNT
    assert report.column_count == 5
    assert report.is_official_layout is True
    assert report.issues == ()
    expected_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert report.sha256 == expected_hash
    assert report.byte_size == fixture.stat().st_size
    assert set(report.label_class_counts) == {
        f"label_{h}" for h in OFFICIAL_FI2010_LABEL_HORIZONS
    }
    for counts in report.label_class_counts.values():
        assert set(counts) <= {"1", "2", "3"}
        assert sum(counts.values()) == 5


def test_inspect_official_fi2010_file_flags_wrong_row_count(tmp_path: Path) -> None:
    short_file = tmp_path / "too_short.txt"
    short_file.write_text("1 2 3\n4 5 6\n", encoding="utf-8")
    report = inspect_official_fi2010_file(short_file)
    assert report.is_official_layout is False
    assert report.row_count == 2
    assert any("149-row" in issue for issue in report.issues)


def test_inspect_official_fi2010_file_flags_ragged_columns(tmp_path: Path) -> None:
    ragged = tmp_path / "ragged.txt"
    ragged.write_text("1 2 3\n4 5\n", encoding="utf-8")
    report = inspect_official_fi2010_file(ragged)
    assert any("column count" in issue for issue in report.issues)


def test_inspect_official_fi2010_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        inspect_official_fi2010_file(tmp_path / "does_not_exist.txt")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def test_convert_official_fi2010_to_csv_round_trips_through_loader(tmp_path: Path) -> None:
    fixture = _write_synthetic_official_file(tmp_path, n_snapshots=8)
    csv_path = tmp_path / "converted.csv"
    report = convert_official_fi2010_to_csv(fixture, csv_path, split_label="train")
    assert isinstance(report, OfficialFI2010ConversionReport)
    assert report.n_samples == 8
    assert report.n_features == (
        OFFICIAL_FI2010_LOB_ROW_COUNT + OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT
    )
    assert report.n_labels == len(OFFICIAL_FI2010_LABEL_HORIZONS)
    assert csv_path.exists()
    assert report.bytes_written == csv_path.stat().st_size

    frame = pd.read_csv(csv_path)
    assert len(frame) == 8
    assert "bid_price_1" in frame.columns
    assert "ask_price_1" in frame.columns
    assert "f_104" in frame.columns
    assert "label_10" in frame.columns
    assert "label_100" in frame.columns
    assert set(frame["split"].unique()) == {"train"}
    assert ((frame["ask_price_1"] - frame["bid_price_1"]) > 0).all()
    assert set(frame["label_10"].astype(int).unique()) <= {1, 2, 3}

    config = FI2010Config(
        path=csv_path,
        timestamp_column=None,
        split_column="split",
        label_columns=["label_10", "label_50", "label_100"],
        price_level_count=OFFICIAL_FI2010_LEVEL_COUNT,
    )
    dataset = load_fi2010(config)
    assert dataset.n_rows == 8
    assert dataset.has_labels
    assert dataset.n_labels == 3


def test_convert_official_fi2010_to_csv_without_split_column(tmp_path: Path) -> None:
    fixture = _write_synthetic_official_file(tmp_path, n_snapshots=4)
    csv_path = tmp_path / "no_split.csv"
    report = convert_official_fi2010_to_csv(fixture, csv_path)
    assert report.split_label is None
    frame = pd.read_csv(csv_path)
    assert "split" not in frame.columns


def test_convert_official_fi2010_to_csv_rejects_wrong_row_count(tmp_path: Path) -> None:
    short = tmp_path / "short.txt"
    short.write_text("1 2\n3 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="149"):
        convert_official_fi2010_to_csv(short, tmp_path / "out.csv")


def test_convert_official_fi2010_to_csv_rejects_invalid_split_label(
    tmp_path: Path,
) -> None:
    fixture = _write_synthetic_official_file(tmp_path)
    with pytest.raises(ValueError):
        convert_official_fi2010_to_csv(
            fixture,
            tmp_path / "out.csv",
            split_label="validation",
        )


def test_convert_official_fi2010_to_csv_overwrite_protection(tmp_path: Path) -> None:
    fixture = _write_synthetic_official_file(tmp_path)
    out = tmp_path / "out.csv"
    convert_official_fi2010_to_csv(fixture, out)
    with pytest.raises(FileExistsError):
        convert_official_fi2010_to_csv(fixture, out)
    convert_official_fi2010_to_csv(fixture, out, overwrite=True)


def test_convert_official_fi2010_to_csv_rejects_directory_output(tmp_path: Path) -> None:
    fixture = _write_synthetic_official_file(tmp_path)
    directory = tmp_path / "out_dir"
    directory.mkdir()
    with pytest.raises(IsADirectoryError):
        convert_official_fi2010_to_csv(fixture, directory, overwrite=True)


def test_convert_official_fi2010_to_csv_rejects_non_finite_value(tmp_path: Path) -> None:
    fixture = _write_synthetic_official_file(tmp_path)
    contents = fixture.read_text(encoding="utf-8").splitlines()
    tokens = contents[0].split()
    tokens[0] = "nan"
    contents[0] = " ".join(tokens)
    fixture.write_text("\n".join(contents) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        convert_official_fi2010_to_csv(fixture, tmp_path / "out.csv")


def test_convert_official_fi2010_to_csv_accepts_official_float_labels(
    tmp_path: Path,
) -> None:
    """Official FI-2010 .txt files store labels as floats like 1.0000000e+00."""
    fixture = _write_synthetic_official_file(tmp_path, n_snapshots=4)
    contents = fixture.read_text(encoding="utf-8").splitlines()
    label_start = OFFICIAL_FI2010_LOB_ROW_COUNT + OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT
    for offset in range(len(OFFICIAL_FI2010_LABEL_HORIZONS)):
        tokens = contents[label_start + offset].split()
        contents[label_start + offset] = " ".join(
            f"{int(token):d}.0000000e+00" for token in tokens
        )
    fixture.write_text("\n".join(contents) + "\n", encoding="utf-8")

    out = tmp_path / "official_format.csv"
    report = convert_official_fi2010_to_csv(fixture, out, split_label="train")
    assert report.n_samples == 4
    frame = pd.read_csv(out)
    assert set(frame["label_10"].astype(int).unique()) <= {1, 2, 3}
    assert set(frame["label_100"].astype(int).unique()) <= {1, 2, 3}


def test_inspect_official_fi2010_file_counts_official_float_labels(
    tmp_path: Path,
) -> None:
    fixture = _write_synthetic_official_file(tmp_path, n_snapshots=5)
    contents = fixture.read_text(encoding="utf-8").splitlines()
    label_start = OFFICIAL_FI2010_LOB_ROW_COUNT + OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT
    for offset in range(len(OFFICIAL_FI2010_LABEL_HORIZONS)):
        tokens = contents[label_start + offset].split()
        contents[label_start + offset] = " ".join(
            f"{int(token):d}.0000000e+00" for token in tokens
        )
    fixture.write_text("\n".join(contents) + "\n", encoding="utf-8")

    report = inspect_official_fi2010_file(fixture)
    assert report.issues == ()
    for counts in report.label_class_counts.values():
        assert set(counts) <= {"1", "2", "3"}
        assert sum(counts.values()) == 5


def test_convert_official_fi2010_to_csv_rejects_unknown_label_value(
    tmp_path: Path,
) -> None:
    fixture = _write_synthetic_official_file(tmp_path)
    contents = fixture.read_text(encoding="utf-8").splitlines()
    label_row_index = (
        OFFICIAL_FI2010_LOB_ROW_COUNT + OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT
    )
    tokens = contents[label_row_index].split()
    tokens[0] = "0"
    contents[label_row_index] = " ".join(tokens)
    fixture.write_text("\n".join(contents) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="FI-2010 class set"):
        convert_official_fi2010_to_csv(fixture, tmp_path / "out.csv")


# ---------------------------------------------------------------------------
# CLI impls
# ---------------------------------------------------------------------------


def test_verify_fi2010_local_cli_succeeds_on_official_layout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _write_synthetic_official_file(tmp_path)
    exit_code = _verify_fi2010_local_impl(data_path=fixture)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "official layout:   True" in captured.out
    assert "issues:            none" in captured.out
    assert "outputs:           not written" in captured.out


def test_verify_fi2010_local_cli_reports_issues_for_unexpected_layout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    short = tmp_path / "short.txt"
    short.write_text("1 2 3\n4 5 6\n", encoding="utf-8")
    exit_code = _verify_fi2010_local_impl(data_path=short)
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "official layout:   False" in captured.out


def test_verify_fi2010_local_cli_reports_missing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = _verify_fi2010_local_impl(data_path=tmp_path / "nope.txt")
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "File not found" in captured.err


def test_convert_fi2010_official_cli_writes_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _write_synthetic_official_file(tmp_path, n_snapshots=3)
    out = tmp_path / "out.csv"
    exit_code = _convert_fi2010_official_impl(
        input_path=fixture,
        output_path=out,
        split_label="test",
        overwrite=False,
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert out.exists()
    assert "samples written:   3" in captured.out
    assert "split column:      test" in captured.out


def test_convert_fi2010_official_cli_reports_missing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = _convert_fi2010_official_impl(
        input_path=tmp_path / "missing.txt",
        output_path=tmp_path / "out.csv",
        split_label=None,
        overwrite=False,
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "File not found" in captured.err


def test_convert_fi2010_official_cli_blocks_overwrite_without_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _write_synthetic_official_file(tmp_path)
    out = tmp_path / "out.csv"
    out.write_text("placeholder\n", encoding="utf-8")
    exit_code = _convert_fi2010_official_impl(
        input_path=fixture,
        output_path=out,
        split_label=None,
        overwrite=False,
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Output already exists" in captured.err
