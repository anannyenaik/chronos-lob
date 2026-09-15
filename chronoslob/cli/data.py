"""Data inspection and preparation command implementations."""

from __future__ import annotations

import sys
from pathlib import Path


def _inspect_experiment_artifacts_impl(*, experiment: Path) -> int:
    """Inspect an experiment directory against the artefact contract."""
    from chronoslob.experiments.artifacts import (
        expected_experiment_artifacts,
        validate_experiment_directory,
    )

    expectations = expected_experiment_artifacts(include_plots=True)
    report = validate_experiment_directory(experiment, include_plots=True)
    expected_by_kind = {expectation.path: expectation for expectation in expectations}

    print("ChronosLOB experiment artefact inspection")
    print(f"  experiment:       {experiment}")
    print(f"  valid:            {'yes' if report.is_valid else 'no'}")
    print(f"  missing required: {len(report.missing_required)}")
    print(f"  optional present: {len(report.present_optional)}")

    print("  required artefacts:")
    for status in report.artefact_statuses:
        if not status.required:
            continue
        state = "present" if status.exists else "missing"
        if status.message.startswith("invalid schema"):
            state = "invalid"
        print(f"    {status.path}: {state}; {status.message}")

    print("  optional artefacts:")
    for status in report.artefact_statuses:
        if status.required:
            continue
        expectation = expected_by_kind.get(status.path)
        if expectation is None:
            candidates = status.path
        else:
            candidates = " or ".join(expectation.candidate_paths)
        state = "present" if status.exists else "missing"
        print(f"    {candidates}: {state}; {status.message}")

    if report.warnings:
        print("  warnings:")
        for warning in report.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:         none")

    print("  training run:     none")
    print("  outputs:          not written")
    print("  network calls:    none performed")
    return 0


def _prepare_fi2010_benchmark_impl(
    *,
    config_path: Path,
    data_path: Path,
    out: Path,
) -> int:
    """Run the local-only FI-2010 benchmark preparation."""
    from chronoslob.data.validation import DataValidationError
    from chronoslob.experiments.fi2010_benchmark import (
        load_benchmark_config,
        prepare_fi2010_benchmark,
    )

    resolved_config_path = Path(config_path)
    try:
        config = load_benchmark_config(resolved_config_path)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"Failed to load benchmark config: {exc}", file=sys.stderr)
        return 1

    try:
        result = prepare_fi2010_benchmark(
            config,
            data_path=Path(data_path),
            output_dir=Path(out),
            config_source_path=resolved_config_path,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except DataValidationError as exc:
        print(f"FI-2010 validation failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"FI-2010 benchmark preparation failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 benchmark preparation")
    print(f"  config:              {resolved_config_path}")
    print(f"  data path:           {result.summary.data_path}")
    print(f"  output directory:    {result.summary.output_dir}")
    print(f"  experiment name:     {result.summary.experiment_name}")
    print(f"  dataset name:        {result.summary.dataset_name}")
    print(f"  task name:           {result.summary.task_name}")
    print(f"  horizon:             {result.summary.horizon}")
    print(f"  split name:          {result.summary.split_name}")
    print(f"  label name:          {result.summary.label_name}")
    print(f"  rows:                {result.split_summary.n_rows}")
    print(f"  train rows:          {result.split_summary.n_train}")
    print(f"  validation rows:     {result.split_summary.n_validation}")
    print(f"  test rows:           {result.split_summary.n_test}")
    print(f"  distinct labels:     {len(result.label_summary.distinct_classes)}")
    print(
        "  fi2010 validation:   "
        f"ok={result.validation_summary.fi2010_validation_ok} "
        f"errors={result.validation_summary.fi2010_error_count} "
        f"warnings={result.validation_summary.fi2010_warning_count}"
    )
    print(
        "  label validation:    "
        f"ok={result.validation_summary.label_validation_ok} "
        f"errors={result.validation_summary.label_error_count} "
        f"warnings={result.validation_summary.label_warning_count}"
    )
    print("  artefacts written:")
    for path in result.written_files:
        print(f"    {path}")
    if result.summary.warnings:
        print("  warnings:")
        for warning in result.summary.warnings:
            print(f"    - {warning}")
    print("  results.json:        not written (preparation only)")
    print("  predictions:         not written (preparation only)")
    print("  network calls:       none performed")
    return 0


def _verify_fi2010_local_impl(*, data_path: Path) -> int:
    """Safely inspect a local FI-2010 file and report its layout."""
    from chronoslob.data.fi2010_official import inspect_official_fi2010_file

    candidate = Path(data_path)
    try:
        report = inspect_official_fi2010_file(candidate)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"FI-2010 verification failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 local verification")
    print(f"  path:              {report.path}")
    print(f"  byte size:         {report.byte_size}")
    print(f"  sha256:            {report.sha256}")
    print(f"  row count:         {report.row_count}")
    print(f"  column count:      {report.column_count}")
    print(f"  official layout:   {report.is_official_layout}")
    print(f"  label horizons:    {list(report.label_horizons)}")
    if report.label_class_counts:
        print("  label class counts:")
        for label_name, counts in report.label_class_counts.items():
            rendered = ", ".join(f"{cls}={count}" for cls, count in counts.items())
            print(f"    {label_name}: {rendered if rendered else '(empty)'}")
    if report.issues:
        print("  issues:")
        for issue in report.issues:
            print(f"    - {issue}")
    else:
        print("  issues:            none")
    print("  network calls:     none performed")
    print("  outputs:           not written")
    if report.is_official_layout:
        print(
            "  next step:         run convert-fi2010-official to produce a loader-ready CSV",
        )
    return 0 if not report.issues else 1


def _convert_fi2010_official_impl(
    *,
    input_path: Path,
    output_path: Path,
    split_label: str | None,
    overwrite: bool,
) -> int:
    """Convert a single official FI-2010 ``.txt`` matrix into a loader-ready CSV."""
    from chronoslob.data.fi2010_official import convert_official_fi2010_to_csv

    try:
        report = convert_official_fi2010_to_csv(
            input_path=Path(input_path),
            output_path=Path(output_path),
            split_label=split_label,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Output already exists: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, IsADirectoryError) as exc:
        print(f"FI-2010 conversion failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 official-format conversion")
    print(f"  input:             {report.input_path}")
    print(f"  output:            {report.output_path}")
    print(f"  samples written:   {report.n_samples}")
    print(f"  feature columns:   {report.n_features}")
    print(f"  label columns:     {report.n_labels}")
    print(f"  label horizons:    {list(report.label_horizons)}")
    if report.split_label is not None:
        print(f"  split column:      {report.split_label}")
    else:
        print("  split column:      not written")
    print(f"  bytes written:     {report.bytes_written}")
    print("  network calls:     none performed")
    return 0


def _is_synthetic_fixture_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "tests" in parts and "fixtures" in parts


def _print_synthetic_fixture_warning(path: Path) -> None:
    if _is_synthetic_fixture_path(path):
        print("WARNING: event log path is a synthetic fixture; outputs are not real market data.")


def _inspect_event_log_impl(path: Path) -> int:
    """Inspect a local canonical event log without writing outputs."""
    from chronoslob.data.manifests import create_event_log_manifest

    path = Path(path)
    _print_synthetic_fixture_warning(path)
    try:
        manifest = create_event_log_manifest(path)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError) as exc:
        print(f"Failed to inspect event log: {exc}", file=sys.stderr)
        return 1

    symbols = ", ".join(manifest.symbols) if manifest.symbols else "none"
    start = manifest.start_timestamp.isoformat() if manifest.start_timestamp is not None else "n/a"
    end = manifest.end_timestamp.isoformat() if manifest.end_timestamp is not None else "n/a"
    seq_range = (
        f"{manifest.min_sequence_id}..{manifest.max_sequence_id}"
        if manifest.min_sequence_id is not None and manifest.max_sequence_id is not None
        else "n/a"
    )

    print("ChronosLOB event log inspection")
    print(f"  path:             {path}")
    print(f"  records:          {manifest.n_records}")
    print(f"  book events:      {manifest.n_book_events}")
    print(f"  snapshots:        {manifest.n_snapshots}")
    print(f"  symbols:          {symbols}")
    print(f"  timestamp range:  {start} to {end}")
    print(f"  sequence range:   {seq_range}")
    print(f"  sha256 prefix:    {manifest.sha256[:12]}")
    print("  outputs:          not written")
    print("  network calls:    none performed")
    return 0


def _inspect_event_tokens_impl(
    path: Path,
    *,
    symbol: str | None = None,
    window_length: int = 8,
    max_levels_per_side: int = 2,
    include_eos: bool = False,
) -> int:
    """Tokenise a canonical event log and print a read-only summary."""
    from chronoslob.models.tokenisation import (
        TokenisationConfig,
        tokenise_event_log,
    )
    from chronoslob.training.token_datasets import (
        TokenWindowConfig,
        build_token_window_indices,
    )

    path = Path(path)
    _print_synthetic_fixture_warning(path)
    try:
        config = TokenisationConfig(
            max_levels_per_side=max_levels_per_side,
            include_eos=include_eos,
        )
        sequence = tokenise_event_log(path, config, symbol=symbol)
        windows = build_token_window_indices(
            sequence,
            TokenWindowConfig(window_length=window_length),
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError) as exc:
        print(f"Failed to inspect event tokens: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB event-token inspection")
    print(f"  path:                    {path}")
    print(f"  symbol filter:           {symbol if symbol is not None else 'none'}")
    print(f"  input records:           {sequence.input_record_count}")
    print(f"  tokenised records:       {len(sequence.records)}")
    print(f"  token windows:           {len(windows)}")
    print(f"  window length:           {window_length}")
    print(f"  snapshot-derived tokens: {'yes' if sequence.has_snapshot_derived_tokens else 'no'}")
    print("  vocabulary sizes:")
    for field_name, size in sequence.field_sizes.items():
        print(f"    {field_name}: {size}")
    print("  first token ids:")
    for record in sequence.records[:5]:
        print(f"    pos={record.position} ids={record.field_id_mapping()}")
    print("  outputs:                 not written")
    print("  network calls:           none performed")
    return 0


def _event_log_to_features_impl(path: Path) -> int:
    """Build replay-derived features from a local event log without writing."""
    from chronoslob.book.event_replay import replay_event_log_to_feature_frame
    from chronoslob.data.event_store import read_event_log_jsonl
    from chronoslob.features.pipeline import validate_feature_frame

    path = Path(path)
    _print_synthetic_fixture_warning(path)
    try:
        records = read_event_log_jsonl(path)
        frame = replay_event_log_to_feature_frame(records)
        validation = validate_feature_frame(frame)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError) as exc:
        print(f"Failed to replay event log to features: {exc}", file=sys.stderr)
        return 1

    feature_columns = [
        column for column in frame.columns if column not in {"timestamp", "symbol", "split"}
    ]

    print("ChronosLOB event-log-to-features inspection")
    print(f"  path:                {path}")
    print(f"  rows:                {len(frame)}")
    print(f"  feature columns:     {len(feature_columns)}")
    print(f"  synthetic_time:      {frame.attrs.get('synthetic_time', False)}")
    print(f"  skipped time feats:  {frame.attrs.get('skipped_time_features', False)}")
    print(f"  validation ok:       {validation.ok}")
    print(f"  validation errors:   {validation.error_count}")
    print(f"  validation warnings: {validation.warning_count}")
    print("  outputs:             not written")
    print("  network calls:       none performed")
    return 0


def _inspect_fi2010_impl(
    path: Path,
    *,
    timestamp_column: str | None = "timestamp",
    split_column: str | None = "split",
    price_level_count: int = 2,
) -> int:
    """Load an FI-2010 file and print a short data-quality summary.

    The function is intentionally read-only: it does not train, transform
    or persist anything. ``timestamp_column`` and ``split_column`` default
    to the names used by the bundled test fixture, but pass ``None`` to
    disable either when running against the canonical FI-2010 matrix.
    """
    from chronoslob.data.fi2010 import FI2010Config, load_fi2010
    from chronoslob.data.validation import validate_fi2010_dataset

    try:
        config = FI2010Config(
            path=path,
            timestamp_column=timestamp_column,
            split_column=split_column,
            price_level_count=price_level_count,
        )
        dataset = load_fi2010(config)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"Failed to load FI-2010 file: {exc}", file=sys.stderr)
        return 1

    summary = dataset.describe()
    validation = validate_fi2010_dataset(dataset)

    print("ChronosLOB FI-2010 inspection")
    print(f"  path:         {path}")
    print(f"  rows:         {dataset.n_rows}")
    print(f"  features:     {dataset.n_features}")
    print(f"  labels:       {dataset.n_labels}")
    print(f"  has labels:   {dataset.has_labels}")
    print(f"  has split:    {summary['has_split_column']}")
    print(f"  has ts col:   {summary['has_timestamp_column']}")
    print(f"  ok:           {validation.ok}")
    print(f"  errors:       {validation.error_count}")
    print(f"  warnings:     {validation.warning_count}")
    return 0


def _inspect_features_fi2010_impl(
    path: Path,
    *,
    timestamp_column: str | None = "timestamp",
    split_column: str | None = "split",
    label_columns: list[str] | None = None,
    price_level_count: int = 2,
    allow_synthetic_timestamps_for_time_features: bool = False,
) -> int:
    """Load an FI-2010 file, build features and print a short summary.

    The function is read-only and does not write anything. It validates
    the resulting feature frame and prints row count, feature count and
    a small list of feature columns.
    """
    from chronoslob.data.fi2010 import FI2010Config, load_fi2010
    from chronoslob.features.pipeline import (
        FeaturePipelineConfig,
        build_feature_frame_from_fi2010,
        validate_feature_frame,
    )

    resolved_labels = (
        list(label_columns) if label_columns is not None else ["label_10", "label_50", "label_100"]
    )
    try:
        config = FI2010Config(
            path=path,
            timestamp_column=timestamp_column,
            split_column=split_column,
            label_columns=resolved_labels,
            price_level_count=price_level_count,
        )
        dataset = load_fi2010(config)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"Failed to load FI-2010 file: {exc}", file=sys.stderr)
        return 1

    pipeline_config = FeaturePipelineConfig(
        allow_synthetic_timestamps_for_time_features=(allow_synthetic_timestamps_for_time_features)
    )
    try:
        frame = build_feature_frame_from_fi2010(dataset, pipeline_config)
    except (ValueError, TypeError) as exc:
        print(f"Failed to build feature frame: {exc}", file=sys.stderr)
        return 1

    feature_columns = [
        column for column in frame.columns if column not in {"timestamp", "symbol", "split"}
    ]
    validation = validate_feature_frame(frame)

    print("ChronosLOB FI-2010 feature inspection")
    print(f"  path:                {path}")
    print(f"  rows:                {len(frame)}")
    print(f"  feature columns:     {len(feature_columns)}")
    print(f"  synthetic_time:      {frame.attrs.get('synthetic_time', False)}")
    print(f"  skipped time feats:  {frame.attrs.get('skipped_time_features', False)}")
    print(f"  validation ok:       {validation.ok}")
    print(f"  validation errors:   {validation.error_count}")
    print(f"  validation warnings: {validation.warning_count}")
    sample = feature_columns[:10]
    print(f"  sample columns:      {sample}")
    return 0


def _inspect_labels_fi2010_impl(
    path: Path,
    *,
    timestamp_column: str | None = "timestamp",
    split_column: str | None = "split",
    label_columns: list[str] | None = None,
    price_level_count: int = 2,
    prefer_existing_labels: bool = True,
) -> int:
    """Load an FI-2010 file, build or extract labels and print a summary."""
    from chronoslob.data.fi2010 import FI2010Config, load_fi2010
    from chronoslob.labels.pipeline import (
        build_label_frame_from_fi2010,
        validate_label_frame,
    )

    resolved_labels = (
        list(label_columns) if label_columns is not None else ["label_10", "label_50", "label_100"]
    )
    try:
        config = FI2010Config(
            path=path,
            timestamp_column=timestamp_column,
            split_column=split_column,
            label_columns=resolved_labels,
            price_level_count=price_level_count,
        )
        dataset = load_fi2010(config)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"Failed to load FI-2010 file: {exc}", file=sys.stderr)
        return 1

    try:
        frame = build_label_frame_from_fi2010(
            dataset,
            prefer_existing_labels=prefer_existing_labels,
        )
    except (ValueError, TypeError) as exc:
        print(f"Failed to build label frame: {exc}", file=sys.stderr)
        return 1

    non_label_columns = {
        "timestamp",
        "symbol",
        "horizon_start",
        "horizon_end",
        "split",
        "label_source",
    }
    label_cols = [column for column in frame.columns if column not in non_label_columns]
    validation = validate_label_frame(frame)

    print("ChronosLOB FI-2010 label inspection")
    print(f"  path:                {path}")
    print(f"  rows:                {len(frame)}")
    print(f"  label columns:       {len(label_cols)}")
    print(f"  validation ok:       {validation.ok}")
    print(f"  validation errors:   {validation.error_count}")
    print(f"  validation warnings: {validation.warning_count}")
    print(f"  sample columns:      {label_cols[:10]}")
    return 0


def _inspect_split_impl(rows: int) -> int:
    """Build a default temporal split and print partition counts."""
    from chronoslob.training.splitters import temporal_train_validation_test_split

    try:
        split = temporal_train_validation_test_split(rows)
    except (TypeError, ValueError) as exc:
        print(f"Failed to build split: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB temporal split inspection")
    print(f"  rows:        {rows}")
    print(f"  train:       {split.n_train}")
    print(f"  validation:  {split.n_validation}")
    print(f"  test:        {split.n_test}")
    return 0


def _init_run_impl(
    *,
    name: str,
    phase: str,
    seed: int,
    root: Path,
    config_path: Path | None = None,
    notes: str | None = None,
) -> int:
    """Initialise a metadata-only experiment run directory."""
    from chronoslob.training.experiment import initialise_experiment_run

    try:
        metadata, run_path = initialise_experiment_run(
            root=root,
            run_name=name,
            phase=phase,
            seed=seed,
            config_path=config_path,
            notes=notes,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"Failed to initialise run: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB experiment run initialised")
    print(f"  run id:       {metadata.run_id}")
    print(f"  run name:     {metadata.run_name}")
    print(f"  run group:    {metadata.phase}")
    print(f"  seed:         {metadata.seed}")
    print(f"  output path:  {run_path}")
    print("  metrics:      none")
    return 0


def _inspect_baselines_impl() -> int:
    """Print supported classical baseline model types without training."""
    from chronoslob.models.baselines import SUPPORTED_BASELINE_MODEL_TYPES

    print("ChronosLOB supported classical baselines")
    for model_type in SUPPORTED_BASELINE_MODEL_TYPES:
        print(f"  - {model_type}")
    print("No training was run.")
    return 0


def _run_baseline_smoke_impl(
    path: Path,
    *,
    write_outputs: bool = False,
    output_root: Path = Path("runs"),
) -> int:
    """Run a tiny synthetic-fixture baseline smoke experiment."""
    from chronoslob.data.fi2010 import FI2010Config, load_fi2010
    from chronoslob.features.pipeline import (
        FeaturePipelineConfig,
        build_feature_frame_from_fi2010,
    )
    from chronoslob.labels.pipeline import build_label_frame_from_fi2010
    from chronoslob.training.baseline_experiment import (
        BaselineExperimentConfig,
        create_default_baseline_configs,
        run_baseline_experiment,
    )

    label_columns = ["label_10", "label_50", "label_100"]
    try:
        dataset = load_fi2010(
            FI2010Config(
                path=path,
                timestamp_column="timestamp",
                split_column="split",
                label_columns=label_columns,
                price_level_count=2,
            )
        )
        feature_frame = build_feature_frame_from_fi2010(
            dataset,
            FeaturePipelineConfig(
                include_order_flow=False,
                include_volatility=False,
            ),
        )
        labels = build_label_frame_from_fi2010(
            dataset,
            prefer_existing_labels=True,
        )
        label_frame = labels.loc[:, ["timestamp", "symbol", "label_10"]]
        config = BaselineExperimentConfig(
            run_name="synthetic-fi2010-baseline-smoke",
            seed=42,
            target_column="label_10",
            models=create_default_baseline_configs(seed=42),
        )
        result = run_baseline_experiment(
            feature_frame,
            label_frame,
            config,
            output_root=output_root,
            write_outputs=write_outputs,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError, OSError) as exc:
        print(f"Baseline smoke failed: {exc}", file=sys.stderr)
        return 1

    print("Synthetic fixture smoke test only; not benchmark performance.")
    print(f"  path:          {path}")
    print(f"  target:        {result['target_column']}")
    print(f"  train rows:    {result['split_sizes']['train']}")
    print(f"  validation:    {result['split_sizes']['validation']}")
    print(f"  test rows:     {result['split_sizes']['test']}")
    print("  validation metrics:")
    for model_result in result["models"]:
        metrics = model_result["validation"]["metrics"]
        print(
            "    "
            f"{model_result['name']}: "
            f"accuracy={metrics['accuracy']:.6f}, "
            f"macro_f1={metrics['macro_f1']:.6f}"
        )
    if write_outputs:
        print(f"  output path:   {result['output_path']}")
    else:
        print("  outputs:       not written")
    return 0


def _inspect_torch_dataset_impl(
    path: Path,
    *,
    lookback: int = 2,
    batch_size: int = 4,
    target_column: str = "label_10",
    timestamp_column: str | None = "timestamp",
    split_column: str | None = "split",
    price_level_count: int = 2,
    train_fraction: float = 0.5,
    validation_fraction: float = 0.34,
    test_fraction: float = 0.16,
) -> int:
    """Build a tiny sequence DataLoader from an FI-2010 fixture and summarise.

    The command is read-only: it does not train, write checkpoints or
    persist any outputs. It is intended only for smoke-testing the
    sequence data layer on the bundled synthetic fixture.
    """
    try:
        from chronoslob.training.datasets import torch_is_available
    except ImportError as exc:  # pragma: no cover - defensive
        print(f"PyTorch is unavailable: {exc}", file=sys.stderr)
        return 3

    if not torch_is_available():
        print(
            "PyTorch is not installed. Install the 'torch' optional "
            "dependency: pip install -e '.[torch]'",
            file=sys.stderr,
        )
        return 3

    from chronoslob.data.fi2010 import FI2010Config, load_fi2010
    from chronoslob.features.pipeline import (
        FeaturePipelineConfig,
        build_feature_frame_from_fi2010,
    )
    from chronoslob.labels.pipeline import build_label_frame_from_fi2010
    from chronoslob.training.dataloaders import (
        DataLoaderConfig,
        build_dataloaders_for_split,
    )
    from chronoslob.training.datasets import SequenceWindowConfig
    from chronoslob.training.splitters import (
        TemporalSplitConfig,
        temporal_train_validation_test_split,
    )

    label_columns = ["label_10", "label_50", "label_100"]
    try:
        dataset = load_fi2010(
            FI2010Config(
                path=path,
                timestamp_column=timestamp_column,
                split_column=split_column,
                label_columns=label_columns,
                price_level_count=price_level_count,
            )
        )
        feature_frame = build_feature_frame_from_fi2010(
            dataset,
            FeaturePipelineConfig(
                include_order_flow=False,
                include_volatility=False,
            ),
        )
        labels = build_label_frame_from_fi2010(
            dataset,
            prefer_existing_labels=True,
        )
        if target_column not in labels.columns:
            print(
                f"target column {target_column!r} is missing from label frame",
                file=sys.stderr,
            )
            return 1
        label_frame = labels.loc[:, ["timestamp", "symbol", target_column]]
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"Failed to prepare frames: {exc}", file=sys.stderr)
        return 1

    try:
        from chronoslob.models.preprocessing import align_feature_label_frames
        from chronoslob.training.datasets import encode_target_values

        aligned = align_feature_label_frames(feature_frame, label_frame)
        split = temporal_train_validation_test_split(
            len(aligned),
            TemporalSplitConfig(
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                min_train_size=1,
                min_validation_size=1,
                min_test_size=0,
            ),
        )
        sequence_config = SequenceWindowConfig(
            lookback=lookback,
            target_column=target_column,
        )
        loader_config = DataLoaderConfig(batch_size=batch_size, shuffle=False)
        # Synthetic fixtures are too small for train to see every class; build the
        # full-frame class mapping so the smoke command demonstrates the data
        # layer end to end. Real experiments should rely on train-only fitting.
        _, full_mapping = encode_target_values(aligned.loc[:, target_column].tolist())
        loaders = build_dataloaders_for_split(
            feature_frame,
            label_frame,
            split,
            sequence_config,
            loader_config,
            class_to_index=full_mapping,
        )
    except (ValueError, TypeError, IndexError) as exc:
        print(f"Failed to build sequence loaders: {exc}", file=sys.stderr)
        return 1

    train_loader = loaders["train"]
    first_batch = next(iter(train_loader))
    train_dataset = train_loader.dataset

    print("Synthetic fixture smoke test only; not benchmark performance.")
    print(f"  path:             {path}")
    print(f"  target column:    {target_column}")
    print(f"  lookback:         {lookback}")
    print(f"  batch size:       {batch_size}")
    print(f"  train samples:    {len(train_dataset)}")
    print(f"  validation:       {len(loaders['validation'].dataset)}")
    if "test" in loaders:
        print(f"  test samples:     {len(loaders['test'].dataset)}")
    else:
        print("  test samples:     0 (no test windows fit)")
    print(f"  feature count:    {train_dataset.n_features}")
    print(f"  batch x shape:    {tuple(first_batch['x'].shape)}")
    print(f"  batch y shape:    {tuple(first_batch['y'].shape)}")
    print(f"  class mapping:    {train_dataset.class_to_index}")
    return 0
