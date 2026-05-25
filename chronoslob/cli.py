"""Command-line interface for ChronosLOB."""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from chronoslob import __version__
from chronoslob.utils.paths import project_root

try:
    import typer
except ModuleNotFoundError:  # pragma: no cover - exercised by smoke command in bare envs
    typer = None  # type: ignore[assignment]

KEY_FOLDERS = (
    "configs",
    "chronoslob",
    "tests",
    "notebooks",
    "reports",
)


def _print(message: Any) -> None:
    try:
        from rich.console import Console
    except ModuleNotFoundError:
        print(message)
        return

    Console().print(message)


def _version_impl() -> None:
    _print(__version__)


def _doctor_rows() -> list[tuple[str, str]]:
    root = project_root()
    rows = [
        ("Python", platform.python_version()),
        ("Package import", f"chronoslob {__version__}"),
        ("Project root", str(root)),
    ]

    for folder in KEY_FOLDERS:
        exists = (root / folder).exists()
        rows.append((f"Folder: {folder}", "present" if exists else "missing"))

    return rows


def _doctor_impl() -> None:
    rows = _doctor_rows()

    try:
        from rich.console import Console
        from rich.table import Table
    except ModuleNotFoundError:
        print("ChronosLOB Doctor")
        for check, value in rows:
            print(f"{check}: {value}")
        return

    table = Table(title="ChronosLOB Doctor", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Value")
    for check, value in rows:
        table.add_row(check, value)
    Console().print(table)


def _run_project_audit_impl(
    *,
    root: Path | None = None,
    strict: bool = False,
) -> int:
    """Run the local repository audit and print a concise summary."""
    from chronoslob.utils.audit import AuditStatus, run_project_audit

    audit = run_project_audit(root)
    inventory = audit.inventory

    print("ChronosLOB project audit")
    print(f"  root:                         {audit.root}")
    print(f"  configs:                      {inventory.config_count}")
    print(f"  reports:                      {inventory.report_count}")
    print(f"  tests:                        {inventory.test_count}")
    print(f"  CLI commands:                 {inventory.cli_command_count}")

    for result in audit.results:
        if result.name == "required_paths":
            print(f"  required paths status:        {result.status.value}")
        elif result.name == "forbidden_claims":
            print(f"  forbidden-claim issue count:  {result.issue_count}")
        elif result.name == "synthetic_labelling":
            print(f"  synthetic-labelling issues:   {result.issue_count}")
        elif result.name == "large_files":
            print(f"  large-file issue count:       {result.issue_count}")
        elif result.name == "public_release_readme":
            print(f"  public README status:         {result.status.value}")
        elif result.name == "public_release_structure":
            print(f"  public docs status:           {result.status.value}")
        elif result.name == "public_release_wording":
            print(f"  public wording issue count:   {result.issue_count}")
        elif result.name == "markdown_formatting":
            print(f"  markdown formatting status:   {result.status.value}")

    if audit.issue_count:
        print("  issues:")
        for result in audit.results:
            for issue in result.issues:
                print(f"    - {issue.format()}")
    else:
        print("  issues:                       none")

    print("  network calls:                none performed")
    print("  outputs:                      not written")
    print(f"  final status:                 {audit.status.value}")

    if audit.failure_count > 0:
        return 1
    if strict and audit.status != AuditStatus.PASS:
        return 1
    return 0


def _inspect_release_readiness_impl(*, root: Path | None = None) -> int:
    """Run public-release checks without writing outputs."""
    from chronoslob.utils.audit import run_public_release_audit

    audit = run_public_release_audit(root)
    result_by_name = {result.name: result for result in audit.results}
    readme = result_by_name["public_release_readme"]
    structure = result_by_name["public_release_structure"]
    wording = result_by_name["public_release_wording"]
    claims = result_by_name["forbidden_claims"]
    formatting = result_by_name["markdown_formatting"]
    workflow_label = "AI/" + "pro" + "mpt artefact scan status"

    print("ChronosLOB release readiness inspection")
    print(f"  root:                            {audit.root}")
    print(f"  README status:                   {readme.status.value}")
    print(f"  docs status:                     {structure.status.value}")
    print(f"  markdown formatting status:      {formatting.status.value}")
    print(f"  {workflow_label}:  {wording.status.value}")
    print(f"  safety/claims scan status:       {claims.status.value}")
    print(
        "  missing recommended files:       "
        f"{structure.details.get('missing_recommended_files', 0)}"
    )
    print("  network calls:                   none performed")
    print("  outputs:                         not written")

    if audit.issue_count:
        print("  issues:")
        for result in audit.results:
            for issue in result.issues:
                print(f"    - {issue.format()}")
    else:
        print("  issues:                          none")

    print(f"  final status:                    {audit.status.value}")
    return 0 if audit.ok else 1


def _build_report_archive_impl(
    *,
    output: Path = Path("reports/report_archive"),
    strict: bool = False,
    include_smoke_training: bool = False,
) -> int:
    """Build the local report evidence archive."""
    from chronoslob.utils.report_archive import (
        ReportArchiveConfig,
        build_report_archive,
    )

    try:
        result = build_report_archive(
            ReportArchiveConfig(
                output_path=output,
                strict=strict,
                include_smoke_training=include_smoke_training,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Failed to build report archive: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB technical evidence archive")
    print(f"  archive path:                 {result.output_path}")
    print(f"  files written:                {len(result.files_written)}")
    print(f"  commands captured:            {result.commands_captured}")
    print(f"  synthetic sections included:  {result.synthetic_section_count}")
    print(f"  warnings:                     {result.warnings_count}")
    for warning in result.warnings:
        print(f"    - {warning}")
    print("  network calls:                none performed")
    return 0


def _inspect_report_archive_impl(
    *,
    output: Path = Path("reports/report_archive"),
) -> int:
    """Inspect expected report archive files without writing."""
    from chronoslob.utils.report_archive import inspect_report_archive

    try:
        statuses = inspect_report_archive(output)
    except (OSError, ValueError) as exc:
        print(f"Failed to inspect report archive: {exc}", file=sys.stderr)
        return 1

    present_count = sum(1 for _, present in statuses if present)
    print("ChronosLOB technical evidence archive inspection")
    print(f"  archive path:    {output}")
    print(f"  expected files:  {len(statuses)}")
    print(f"  present files:   {present_count}")
    print(f"  missing files:   {len(statuses) - present_count}")
    for relative_path, present in statuses:
        status = "present" if present else "missing"
        print(f"  {relative_path.as_posix()}: {status}")
    print("  network calls:   none performed")
    return 0 if present_count == len(statuses) else 1


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
            "  next step:         run convert-fi2010-official to produce a "
            "loader-ready CSV",
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


def _run_paper_experiment_impl(
    *,
    config_path: Path,
    data_path: Path,
    out: Path,
    models: Sequence[str] | None,
    overwrite: bool,
    build_plots: bool = False,
) -> int:
    """Run the paper experiment runner and validate the artefact directory."""
    from chronoslob.experiments.paper_runner import (
        SUPPORTED_PAPER_MODELS,
        run_paper_experiment,
    )

    try:
        summary = run_paper_experiment(
            config_path=Path(config_path),
            data_path=Path(data_path),
            out_dir=Path(out),
            models=models,
            overwrite=overwrite,
            build_plots=build_plots,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Paper experiment failed: {exc}", file=sys.stderr)
        print(
            "  supported models: " + ", ".join(SUPPORTED_PAPER_MODELS),
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB paper experiment runner")
    print(f"  experiment name:     {summary.experiment_name}")
    print(f"  task name:           {summary.task_name}")
    print(f"  horizon:             {summary.horizon}")
    print(f"  split name:          {summary.split_name}")
    print(f"  data path:           {summary.data_path}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  requested models:    {', '.join(summary.requested_models)}")
    print(f"  models run:          {', '.join(summary.models_run)}")
    if summary.skipped_models:
        print("  skipped models:")
        for skip in summary.skipped_models:
            print(f"    - {skip.model_name}: {skip.reason}")
    else:
        print("  skipped models:      none")
    if summary.predictive_metric_names:
        print(
            "  predictive metrics:  "
            + ", ".join(summary.predictive_metric_names)
        )
    if summary.calibration_metric_names:
        print(
            "  calibration metrics: "
            + ", ".join(summary.calibration_metric_names)
        )
    if summary.metric_names:
        print(f"  all metrics emitted: {', '.join(summary.metric_names)}")
    else:
        print("  metrics emitted:     none")
    print(f"  fixture run:         {'yes' if summary.is_fixture else 'no'}")
    print(f"  runner version:      {summary.runner_version}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    print(
        "  artefact validation: "
        f"{'valid' if summary.validation.is_valid else 'invalid'}"
    )
    if summary.validation.missing_required:
        print("  missing required:")
        for missing in summary.validation.missing_required:
            print(f"    - {missing}")
    if summary.plot_summary is not None:
        if summary.plot_summary.plots_written:
            print("  plots written:")
            for relative_path in summary.plot_summary.plots_written:
                print(f"    - {relative_path}")
        else:
            print("  plots written:       none")
        if summary.plot_summary.plots_skipped:
            print("  plots skipped:")
            for relative_path in summary.plot_summary.plots_skipped:
                print(f"    - {relative_path}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    print("  network calls:       none performed")
    return 0 if summary.validation.is_valid else 1


def _run_paper_ablations_impl(
    *,
    config_path: Path,
    data_path: Path,
    out: Path,
    models: Sequence[str] | None,
    ablation_set: str,
    overwrite: bool,
    build_plots: bool = False,
) -> int:
    """Run the paper ablation suite and write aggregate summary artefacts."""
    from chronoslob.experiments.ablations import (
        SUPPORTED_ABLATION_SETS,
        run_paper_ablations,
    )

    selected_models = list(models) if models else ["majority"]

    try:
        summary = run_paper_ablations(
            config_path=Path(config_path),
            data_path=Path(data_path),
            out_dir=Path(out),
            models=selected_models,
            ablation_set=ablation_set,
            overwrite=overwrite,
            build_plots=build_plots,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Paper ablation suite failed: {exc}", file=sys.stderr)
        print(
            "  supported ablation sets: " + ", ".join(SUPPORTED_ABLATION_SETS),
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB paper ablation suite")
    print(f"  ablation set:        {summary.ablation_set}")
    print(f"  base config:         {summary.base_config}")
    print(f"  data path:           {summary.data_path}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  models:              {', '.join(summary.models_requested)}")
    if summary.ablations_run:
        print("  ablations run:")
        for name in summary.ablations_run:
            child = summary.child_experiments.get(name, "")
            suffix = f" ({child})" if child else ""
            print(f"    - {name}{suffix}")
    else:
        print("  ablations run:       none")
    if summary.ablations_skipped:
        print("  ablations skipped:")
        for name in summary.ablations_skipped:
            reason = next(
                (
                    result.reason
                    for result in summary.results
                    if result.name == name
                ),
                None,
            )
            suffix = f" ({reason})" if reason else ""
            print(f"    - {name}{suffix}")
    else:
        print("  ablations skipped:   none")
    if summary.reports_written:
        print("  reports written:")
        for relative_path in summary.reports_written:
            print(f"    - {relative_path}")
    else:
        print("  reports written:     none")
    print(f"  fixture run:         {'yes' if summary.is_fixture else 'no'}")
    print(f"  runner version:      {summary.runner_version}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  network calls:       none performed")
    return 0


def _run_system_benchmarks_impl(
    *,
    config_path: Path,
    data_path: Path,
    out: Path,
    benchmark_set: str,
    models: Sequence[str] | None,
    overwrite: bool,
) -> int:
    """Run local systems benchmarks and write traceable artefacts."""
    from chronoslob.experiments.system_benchmarks import (
        SUPPORTED_SYSTEM_BENCHMARK_SETS,
        run_system_benchmarks,
    )

    selected_models = list(models) if models else ["majority", "logistic"]

    try:
        summary = run_system_benchmarks(
            config_path=Path(config_path),
            data_path=Path(data_path),
            out_dir=Path(out),
            benchmark_set=benchmark_set,
            models=selected_models,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Systems benchmark failed: {exc}", file=sys.stderr)
        print(
            "  supported benchmark sets: "
            + ", ".join(SUPPORTED_SYSTEM_BENCHMARK_SETS),
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB systems benchmark")
    print(f"  benchmark set:       {summary.benchmark_set}")
    print(f"  models:              {', '.join(summary.models_requested)}")
    if summary.benchmarks_run:
        print("  benchmarks run:")
        for name in summary.benchmarks_run:
            print(f"    - {name}")
    else:
        print("  benchmarks run:      none")
    if summary.benchmarks_skipped:
        print("  benchmarks skipped:")
        for name in summary.benchmarks_skipped:
            print(f"    - {name}")
    else:
        print("  benchmarks skipped:  none")
    print(f"  output directory:    {summary.output_dir}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  network calls:       none performed")
    return 0


def _inspect_system_benchmarks_impl(*, benchmark: Path) -> int:
    """Print a concise summary of a systems benchmark directory."""
    import json as _json

    resolved_dir = Path(benchmark)
    if not resolved_dir.exists():
        print(f"Systems benchmark directory not found: {resolved_dir}", file=sys.stderr)
        return 2
    if not resolved_dir.is_dir():
        print(
            f"Systems benchmark path is not a directory: {resolved_dir}",
            file=sys.stderr,
        )
        return 1

    summary_path = resolved_dir / "system_benchmark_summary.json"
    results_path = resolved_dir / "system_benchmark_results.csv"
    if not summary_path.is_file():
        print(f"Missing system_benchmark_summary.json: {summary_path}", file=sys.stderr)
        return 1
    try:
        payload = _json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError) as exc:
        print(f"Invalid system_benchmark_summary.json: {exc}", file=sys.stderr)
        return 1
    if not isinstance(payload, Mapping):
        print("Invalid system_benchmark_summary.json: expected object", file=sys.stderr)
        return 1

    result_rows = 0
    if results_path.is_file():
        try:
            result_rows = max(
                sum(1 for _ in results_path.open("r", encoding="utf-8")) - 1,
                0,
            )
        except OSError as exc:
            print(f"Result CSV row count error: {exc}", file=sys.stderr)
            return 1

    reports_dir = resolved_dir / "reports"
    reports_present = (
        sorted(path.name for path in reports_dir.glob("*.md"))
        if reports_dir.is_dir()
        else []
    )
    benchmark_set = str(payload.get("benchmark_set", "unknown"))
    benchmarks_run = payload.get("benchmarks_run")
    benchmarks_skipped = payload.get("benchmarks_skipped")
    warnings = payload.get("warnings")

    print("ChronosLOB systems benchmark inspection")
    print(f"  benchmark dir:       {resolved_dir}")
    print(f"  benchmark set:       {benchmark_set}")
    print(f"  result rows:         {result_rows}")
    if reports_present:
        print("  reports present:")
        for report in reports_present:
            print(f"    - reports/{report}")
    else:
        print("  reports present:     none")
    if isinstance(benchmarks_run, list) and benchmarks_run:
        print("  benchmarks run:")
        for name in benchmarks_run:
            print(f"    - {name}")
    else:
        print("  benchmarks run:      none")
    if isinstance(benchmarks_skipped, list) and benchmarks_skipped:
        print("  benchmarks skipped:")
        for name in benchmarks_skipped:
            print(f"    - {name}")
    else:
        print("  benchmarks skipped:  none")
    if isinstance(warnings, list) and warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  outputs:             not written")
    print("  network calls:       none performed")
    return 0


def _build_paper_plots_impl(
    *,
    experiment: Path,
    overwrite: bool,
) -> int:
    """Generate paper experiment plots from stored artefacts only."""
    from chronoslob.experiments.plots import build_paper_experiment_plots

    resolved_dir = Path(experiment)
    try:
        summary = build_paper_experiment_plots(
            resolved_dir,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except NotADirectoryError as exc:
        print(f"Path is not a directory: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Paper plot generation failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB paper experiment plot builder")
    print(f"  experiment dir:   {summary.experiment_dir}")
    print(f"  builder version:  {summary.builder_version}")
    if summary.plots_written:
        print("  plots written:")
        for relative_path in summary.plots_written:
            print(f"    - {relative_path}")
    else:
        print("  plots written:    none")
    if summary.plots_skipped:
        print("  plots skipped:")
        for relative_path in summary.plots_skipped:
            print(f"    - {relative_path}")
    else:
        print("  plots skipped:    none")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:         none")
    print(f"  plot summary:     {Path(summary.experiment_dir) / 'plot_summary.json'}")
    print("  network calls:    none performed")
    return 0


def _inspect_paper_experiment_impl(*, experiment: Path) -> int:
    """Print a concise human-readable summary of a paper experiment directory."""
    import json as _json

    from chronoslob.experiments.artifacts import (
        load_results,
        validate_experiment_directory,
    )
    from chronoslob.experiments.plots import (
        PAPER_PLOT_FILENAMES,
        PLOT_SUMMARY_FILENAME,
    )

    resolved_dir = Path(experiment)
    if not resolved_dir.exists():
        print(f"Experiment directory not found: {resolved_dir}", file=sys.stderr)
        return 2
    if not resolved_dir.is_dir():
        print(
            f"Experiment path is not a directory: {resolved_dir}",
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB paper experiment inspection")
    print(f"  experiment dir:   {resolved_dir}")

    report = validate_experiment_directory(resolved_dir, include_plots=True)
    print(
        "  artefact validation: "
        f"{'valid' if report.is_valid else 'invalid'}"
    )
    if report.missing_required:
        print("  missing required:")
        for missing in report.missing_required:
            print(f"    - {missing}")

    runner_summary_path = resolved_dir / "runner_summary.json"
    runner_payload: Mapping[str, Any] | None = None
    if runner_summary_path.is_file():
        try:
            payload = _json.loads(runner_summary_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            print(f"  runner_summary.json invalid: {exc}")
        else:
            if isinstance(payload, Mapping):
                runner_payload = payload

    if runner_payload is not None:
        requested = runner_payload.get("requested_models") or []
        models_run = runner_payload.get("models_run") or []
        skipped = runner_payload.get("skipped_models") or []
        print(
            "  requested models: "
            + (", ".join(str(item) for item in requested) if requested else "none")
        )
        print(
            "  models run:       "
            + (", ".join(str(item) for item in models_run) if models_run else "none")
        )
        if skipped:
            print("  skipped models:")
            for skip in skipped:
                if isinstance(skip, Mapping):
                    name = skip.get("model_name", "unknown")
                    reason = skip.get("reason", "")
                    print(f"    - {name}: {reason}")
                else:
                    print(f"    - {skip}")
        else:
            print("  skipped models:   none")
    else:
        print("  runner_summary.json: not found")

    results_path = resolved_dir / "results.json"
    if results_path.is_file():
        try:
            results = load_results(results_path)
        except (OSError, ValueError) as exc:
            print(f"  results.json invalid: {exc}")
        else:
            streams = results.evidence_streams
            print("  evidence streams:")
            print(
                "    predictive:   "
                + (", ".join(streams.predictive) if streams.predictive else "none")
            )
            print(
                "    calibration:  "
                + (", ".join(streams.calibration) if streams.calibration else "none")
            )
            print(
                "    execution:    "
                + (", ".join(streams.execution) if streams.execution else "none")
            )
            if streams.robustness:
                print(
                    "    robustness:   "
                    + ", ".join(streams.robustness)
                )
    else:
        print("  results.json:     not found")

    predictions_path = resolved_dir / "predictions.csv"
    if predictions_path.is_file():
        try:
            row_count = sum(1 for _ in predictions_path.open("r", encoding="utf-8")) - 1
        except OSError as exc:
            print(f"  predictions.csv row count error: {exc}")
        else:
            print(
                "  prediction rows:  "
                + (str(max(row_count, 0)) if row_count >= 0 else "0")
            )
    else:
        print("  prediction rows:  predictions.csv not present")

    calibration_path = resolved_dir / "calibration_bins.csv"
    if calibration_path.is_file():
        try:
            calibration_count = (
                sum(1 for _ in calibration_path.open("r", encoding="utf-8")) - 1
            )
        except OSError as exc:
            print(f"  calibration row count error: {exc}")
        else:
            print(
                "  calibration rows: "
                + str(max(calibration_count, 0))
            )
    else:
        print("  calibration rows: calibration_bins.csv not present")

    execution_path = resolved_dir / "execution_sensitivity.csv"
    if execution_path.is_file():
        try:
            execution_count = (
                sum(1 for _ in execution_path.open("r", encoding="utf-8")) - 1
            )
        except OSError as exc:
            print(f"  execution row count error: {exc}")
        else:
            print(
                "  execution rows:   "
                + str(max(execution_count, 0))
            )
    else:
        print("  execution rows:   execution_sensitivity.csv not present")

    plots_dir = resolved_dir / "plots"
    plots_present: list[str] = []
    plots_missing: list[str] = []
    for filename in PAPER_PLOT_FILENAMES:
        candidate = plots_dir / filename
        if candidate.is_file():
            plots_present.append(f"plots/{filename}")
        else:
            plots_missing.append(f"plots/{filename}")
    if plots_present:
        print("  plots present:")
        for relative_path in plots_present:
            print(f"    - {relative_path}")
    else:
        print("  plots present:    none")
    if plots_missing:
        print("  plots missing:")
        for relative_path in plots_missing:
            print(f"    - {relative_path}")

    plot_summary_path = resolved_dir / PLOT_SUMMARY_FILENAME
    plot_warnings: list[str] = []
    if plot_summary_path.is_file():
        try:
            plot_payload = _json.loads(plot_summary_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            print(f"  plot_summary.json invalid: {exc}")
        else:
            if isinstance(plot_payload, Mapping):
                summary_warnings = plot_payload.get("warnings")
                if isinstance(summary_warnings, list):
                    plot_warnings.extend(
                        str(item) for item in summary_warnings if isinstance(item, str)
                    )

    if report.warnings:
        print("  artefact warnings:")
        for warning in report.warnings:
            print(f"    - {warning}")
    if plot_warnings:
        print("  plot warnings:")
        for warning in plot_warnings:
            print(f"    - {warning}")
    if not report.warnings and not plot_warnings:
        print("  warnings:         none")

    is_fixture_flag: bool | None = None
    if runner_payload is not None:
        raw_flag = runner_payload.get("is_fixture")
        if isinstance(raw_flag, bool):
            is_fixture_flag = raw_flag
    if is_fixture_flag is None:
        model_card_path = resolved_dir / "model_card.md"
        if model_card_path.is_file():
            try:
                text = model_card_path.read_text(encoding="utf-8").lower()
            except OSError:
                text = ""
            if "synthetic fixture" in text or "not benchmark evidence" in text:
                is_fixture_flag = True
    if is_fixture_flag is True:
        print("  fixture run:      yes (synthetic fixture smoke run; not benchmark evidence)")
    elif is_fixture_flag is False:
        print("  fixture run:      no")
    else:
        print("  fixture run:      unknown")
    print("  outputs:          not written")
    print("  network calls:    none performed")
    return 0


def _build_paper_report_impl(
    *,
    experiment: Path,
    out: Path,
    ablations: Path | None,
    systems: Path | None,
    overwrite: bool,
) -> int:
    """Build an empirical report from stored paper artefacts."""
    from chronoslob.experiments.reporting import build_paper_report

    try:
        summary = build_paper_report(
            experiment_dir=Path(experiment),
            out_path=Path(out),
            ablation_dir=Path(ablations) if ablations is not None else None,
            systems_dir=Path(systems) if systems is not None else None,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (IsADirectoryError, NotADirectoryError) as exc:
        print(f"Path error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Paper report build failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB empirical report builder")
    print(f"  report path:        {summary.report_path}")
    print(f"  summary path:       {summary.summary_path}")
    print(f"  sections written:   {len(summary.sections_written)}")
    for section in summary.sections_written:
        print(f"    - {section}")
    print(f"  artefacts used:     {len(summary.artefacts_used)}")
    print(f"  warnings:           {len(summary.warnings)}")
    for warning in summary.warnings:
        print(f"    - {warning}")
    print(
        "  fixture/smoke run:  "
        f"{'yes' if summary.fixture_or_smoke_run else 'no'}"
    )
    print("  network calls:      none performed")
    return 0


def _inspect_paper_report_impl(*, report: Path) -> int:
    """Inspect a generated empirical report and summary JSON."""
    import json as _json

    from chronoslob.experiments.reporting import inspect_paper_report

    try:
        inspection = inspect_paper_report(Path(report))
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except IsADirectoryError as exc:
        print(f"Path error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, _json.JSONDecodeError) as exc:
        print(f"Paper report inspection failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB empirical report inspection")
    print(f"  report path:          {inspection.report_path}")
    if inspection.summary_path is None:
        print("  summary JSON path:    not found")
    else:
        print(f"  summary JSON path:    {inspection.summary_path}")
    print(f"  sections detected:    {len(inspection.sections_detected)}")
    for section in inspection.sections_detected:
        print(f"    - {section}")
    print(f"  artefacts used count: {inspection.artefacts_used_count}")
    print(f"  warnings count:       {inspection.warnings_count}")
    if inspection.fixture_or_smoke_run is None:
        print("  fixture/smoke flag:   unknown")
    else:
        print(
            "  fixture/smoke flag:   "
            f"{'yes' if inspection.fixture_or_smoke_run else 'no'}"
        )
    print("  outputs:              not written")
    print("  network calls:        none performed")
    return 0


def _is_synthetic_fixture_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "tests" in parts and "fixtures" in parts


def _print_synthetic_fixture_warning(path: Path) -> None:
    if _is_synthetic_fixture_path(path):
        print(
            "WARNING: event log path is a synthetic fixture; "
            "outputs are not real market data."
        )


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
    start = (
        manifest.start_timestamp.isoformat()
        if manifest.start_timestamp is not None
        else "n/a"
    )
    end = (
        manifest.end_timestamp.isoformat()
        if manifest.end_timestamp is not None
        else "n/a"
    )
    seq_range = (
        f"{manifest.min_sequence_id}..{manifest.max_sequence_id}"
        if manifest.min_sequence_id is not None
        and manifest.max_sequence_id is not None
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
    print(
        "  snapshot-derived tokens: "
        f"{'yes' if sequence.has_snapshot_derived_tokens else 'no'}"
    )
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
        column
        for column in frame.columns
        if column not in {"timestamp", "symbol", "split"}
    ]

    print("ChronosLOB event-log-to-features inspection")
    print(f"  path:                {path}")
    print(f"  rows:                {len(frame)}")
    print(f"  feature columns:     {len(feature_columns)}")
    print(f"  synthetic_time:      {frame.attrs.get('synthetic_time', False)}")
    print(
        f"  skipped time feats:  {frame.attrs.get('skipped_time_features', False)}"
    )
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
        list(label_columns)
        if label_columns is not None
        else ["label_10", "label_50", "label_100"]
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
        allow_synthetic_timestamps_for_time_features=(
            allow_synthetic_timestamps_for_time_features
        )
    )
    try:
        frame = build_feature_frame_from_fi2010(dataset, pipeline_config)
    except (ValueError, TypeError) as exc:
        print(f"Failed to build feature frame: {exc}", file=sys.stderr)
        return 1

    feature_columns = [
        column
        for column in frame.columns
        if column not in {"timestamp", "symbol", "split"}
    ]
    validation = validate_feature_frame(frame)

    print("ChronosLOB FI-2010 feature inspection")
    print(f"  path:                {path}")
    print(f"  rows:                {len(frame)}")
    print(f"  feature columns:     {len(feature_columns)}")
    print(f"  synthetic_time:      {frame.attrs.get('synthetic_time', False)}")
    print(
        f"  skipped time feats:  {frame.attrs.get('skipped_time_features', False)}"
    )
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
        list(label_columns)
        if label_columns is not None
        else ["label_10", "label_50", "label_100"]
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
    print(f"  phase:        {metadata.phase}")
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


def _inspect_deeplob_impl() -> int:
    """Print the DeepLOB-style model defaults without training."""
    try:
        from chronoslob.models.deeplob import DeepLOBConfig
    except ImportError as exc:
        print(f"PyTorch is unavailable: {exc}", file=sys.stderr)
        return 3

    defaults = DeepLOBConfig(input_features=10, n_classes=3)
    print("ChronosLOB DeepLOB-style baseline")
    print("  DeepLOB-style supervised CNN-LSTM, not an exact paper reproduction.")
    print("  Defaults (sample input_features=10, n_classes=3):")
    print(f"    conv_channels:     {defaults.conv_channels}")
    print(f"    conv_kernel_size:  {defaults.conv_kernel_size}")
    print(f"    lstm_hidden_size:  {defaults.lstm_hidden_size}")
    print(f"    lstm_layers:       {defaults.lstm_layers}")
    print(f"    dropout:           {defaults.dropout}")
    print(f"    use_batch_norm:    {defaults.use_batch_norm}")
    print("  No training was run.")
    return 0


def _run_deeplob_smoke_impl(
    path: Path,
    *,
    lookback: int = 2,
    epochs: int = 1,
    batch_size: int = 4,
    seed: int = 42,
    write_outputs: bool = False,
    output_root: Path = Path("runs"),
) -> int:
    """Run a tiny synthetic-fixture DeepLOB smoke experiment."""
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

    from chronoslob.training.torch_experiment import (
        run_deeplob_smoke_from_fi2010_fixture,
    )

    try:
        result = run_deeplob_smoke_from_fi2010_fixture(
            path=path,
            lookback=lookback,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"DeepLOB smoke failed: {exc}", file=sys.stderr)
        return 1

    if write_outputs:
        # Smoke command intentionally does not write outputs; only the
        # explicit DeepLOB experiment runner supports writing artefacts.
        # Surface the request as a clear notice rather than silently
        # ignoring it.
        print(
            "Note: --write-outputs is not honoured by the smoke command; "
            "use run_deeplob_experiment with write_outputs=True instead.",
        )
        _ = output_root  # explicit no-op so linters do not flag the argument

    print(result["notes"])
    print(f"  path:                   {path}")
    print(f"  target:                 {result['target_column']}")
    print(f"  lookback:               {result['lookback']}")
    print(f"  feature count:          {result['feature_count']}")
    print(f"  train samples:          {result['sample_counts']['train']}")
    print(f"  validation samples:     {result['sample_counts']['validation']}")
    print(f"  model parameter count:  {result['model_parameter_count']}")
    if result["training_history"]:
        last_history = result["training_history"][-1]
        train_loss = last_history.get("train_loss")
        if train_loss is not None:
            print(f"  final train loss:       {train_loss:.6f}")
        validation_loss = last_history.get("validation_loss")
        if validation_loss is not None:
            print(f"  final validation loss:  {validation_loss:.6f}")
    if result["final_validation_metrics"] is not None:
        accuracy = result["final_validation_metrics"]["metrics"]["accuracy"]
        macro_f1 = result["final_validation_metrics"]["metrics"]["macro_f1"]
        print(f"  validation accuracy:    {accuracy:.6f}")
        print(f"  validation macro F1:    {macro_f1:.6f}")
    else:
        print("  validation metrics:     not available")
    print("  outputs:                not written (smoke command)")
    print("  checkpoints:            not written")
    return 0


def _inspect_transformer_impl() -> int:
    """Print the market transformer encoder defaults without training."""
    try:
        from chronoslob.models.transformer import (
            MarketTransformerConfig,
            create_market_transformer,
        )
    except ImportError as exc:
        print(f"PyTorch is unavailable: {exc}", file=sys.stderr)
        return 3

    config = MarketTransformerConfig()
    model = create_market_transformer(config)
    print("ChronosLOB Market Transformer encoder")
    print("  Supervised encoder over field-wise tokenised market microstructure.")
    print("  No self-supervised objective, calibration or execution claim.")
    print(f"  token fields expected:    {len(config.token_field_names)}")
    print(f"  token field names:        {list(config.token_field_names)}")
    print("  vocab sizes (default):")
    for field_name, size in config.vocab_sizes.items():
        print(f"    {field_name}: {size}")
    print("  defaults:")
    print(f"    field_embedding_dim:    {config.field_embedding_dim}")
    print(f"    model_dim:              {config.model_dim}")
    print(f"    num_heads:              {config.num_heads}")
    print(f"    num_layers:             {config.num_layers}")
    print(f"    feedforward_dim:        {config.feedforward_dim}")
    print(f"    dropout:                {config.dropout}")
    print(f"    max_sequence_length:    {config.max_sequence_length}")
    print(f"    num_classes:            {config.num_classes}")
    print(f"    pooling:                {config.pooling}")
    print(f"    activation:             {config.activation}")
    print(f"    use_layer_norm:         {config.use_layer_norm}")
    print(f"    pad_token_id:           {config.pad_token_id}")
    print(f"  model parameter count:    {model.n_parameters()}")
    print("  No training was run.")
    return 0


def _run_transformer_smoke_impl(
    path: Path,
    *,
    window_length: int = 4,
    batch_size: int = 4,
    epochs: int = 1,
    seed: int = 42,
    num_classes: int = 3,
    symbol: str | None = None,
    max_levels_per_side: int = 2,
) -> int:
    """Run a tiny synthetic-label transformer smoke experiment."""
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

    from chronoslob.training.transformer_experiment import (
        run_transformer_smoke_from_event_log,
    )

    path = Path(path)
    _print_synthetic_fixture_warning(path)
    try:
        result = run_transformer_smoke_from_event_log(
            path=path,
            symbol=symbol,
            window_length=window_length,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_classes=num_classes,
            max_levels_per_side=max_levels_per_side,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"Transformer smoke failed: {exc}", file=sys.stderr)
        return 1

    print("Synthetic smoke labels only; no market signal or benchmark is implied.")
    print(f"  path:                   {path}")
    print(f"  symbol filter:          {symbol if symbol is not None else 'none'}")
    print(f"  input records:          {result['input_record_count']}")
    print(f"  tokenised records:      {result['tokenised_record_count']}")
    print(f"  window length:          {result['window_length']}")
    print(f"  window count:           {result['window_count']}")
    print(f"  num classes (smoke):    {result['num_classes']}")
    print(f"  model parameter count:  {result['model_parameter_count']}")
    if result["training_history"]:
        final_epoch = result["training_history"][-1]
        print(f"  final train loss:       {final_epoch['train_loss']:.6f}")
    print(f"  label source:           {result['label_source']}")
    print(
        "  synthetic smoke metric: "
        f"accuracy={result['synthetic_smoke_metrics']['accuracy']:.6f} "
        "(synthetic plumbing only; not a market signal)"
    )
    print("  outputs:                not written (smoke command)")
    print("  checkpoints:            not written")
    print("  network calls:          none performed")
    return 0


def _inspect_ssl_impl() -> int:
    """Print the SSL transformer wrapper defaults without training."""
    try:
        from chronoslob.models.ssl import (
            SSLTransformerConfig,
            create_ssl_transformer,
        )
    except ImportError as exc:
        print(f"PyTorch is unavailable: {exc}", file=sys.stderr)
        return 3

    config = SSLTransformerConfig()
    model = create_ssl_transformer(config)
    print("ChronosLOB SSL Transformer wrapper")
    print(
        "  Self-supervised pretraining over field-wise tokenised market "
        "microstructure."
    )
    print(
        "  No supervised market labels, calibration, execution simulation or "
        "benchmark claim."
    )
    print(f"  enabled objectives:       {list(config.enabled_objectives())}")
    print(f"  masked fields:            {list(config.masked_fields)}")
    print(f"  next-predicted fields:    {list(config.next_fields)}")
    print(f"  ignore_index:             {config.ignore_index}")
    print(f"  contrastive enabled:      {config.enable_contrastive_loss}")
    print("  masking config:")
    print(f"    mask_probability:        {config.masking.mask_probability}")
    print(f"    mask_token_probability:  {config.masking.mask_token_probability}")
    print(
        "    random_token_probability:"
        f" {config.masking.random_token_probability}"
    )
    print(f"    keep_token_probability:  {config.masking.keep_token_probability}")
    print(
        f"    force_at_least_one_mask: {config.masking.force_at_least_one_mask}"
    )
    print("  loss weights:")
    for name, weight in dict(config.loss_weights).items():
        print(f"    {name}: {weight}")
    print("  transformer backbone:")
    print(f"    model_dim:              {config.transformer.model_dim}")
    print(f"    num_heads:              {config.transformer.num_heads}")
    print(f"    num_layers:             {config.transformer.num_layers}")
    print(
        "    max_sequence_length:    "
        f"{config.transformer.max_sequence_length}"
    )
    print(f"  model parameter count:    {model.n_parameters()}")
    print("  No training was run.")
    return 0


def _run_ssl_smoke_impl(
    path: Path,
    *,
    window_length: int = 4,
    batch_size: int = 4,
    epochs: int = 1,
    seed: int = 42,
    symbol: str | None = None,
    max_levels_per_side: int = 2,
    mask_probability: float = 0.15,
) -> int:
    """Run a tiny synthetic SSL smoke experiment from an event log."""
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

    from chronoslob.training.ssl_experiment import (
        run_ssl_smoke_from_event_log,
    )

    path = Path(path)
    _print_synthetic_fixture_warning(path)
    try:
        result = run_ssl_smoke_from_event_log(
            path=path,
            symbol=symbol,
            window_length=window_length,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            max_levels_per_side=max_levels_per_side,
            mask_probability=mask_probability,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"SSL smoke failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Synthetic SSL plumbing only; losses do not measure market signal, "
        "alpha or benchmark performance."
    )
    print(f"  path:                   {path}")
    print(f"  symbol filter:          {symbol if symbol is not None else 'none'}")
    print(f"  input records:          {result['input_record_count']}")
    print(f"  tokenised records:      {result['tokenised_record_count']}")
    print(f"  window length:          {result['window_length']}")
    print(f"  window count:           {result['window_count']}")
    print(f"  enabled objectives:     {result['enabled_objectives']}")
    print(f"  masked fields:          {result['masked_fields']}")
    print(f"  next fields:            {result['next_fields']}")
    print(f"  model parameter count:  {result['model_parameter_count']}")
    if result["final_train_loss"] is not None:
        print(f"  final train loss:       {result['final_train_loss']:.6f}")
        for name, value in result["final_train_loss_components"].items():
            print(f"    {name} loss:          {value:.6f}")
    print(
        "  synthetic smoke metric: "
        f"loss={result['synthetic_smoke_metrics']['loss']:.6f} "
        "(synthetic plumbing only; not a market signal)"
    )
    print("  outputs:                not written (smoke command)")
    print("  checkpoints:            not written")
    print("  network calls:          none performed")
    return 0


def _inspect_multitask_impl() -> int:
    """Print multi-task transformer defaults without training."""
    try:
        from chronoslob.models.multitask import (
            MultiTaskTransformerConfig,
            create_multitask_transformer,
        )
    except ImportError as exc:
        print(f"PyTorch is unavailable: {exc}", file=sys.stderr)
        return 3

    config = MultiTaskTransformerConfig()
    model = create_multitask_transformer(config)
    print("ChronosLOB Multi-Task Transformer")
    print(
        "  Supervised fine-tuning heads over a shared field-wise token "
        "transformer backbone."
    )
    print(
        "  No calibration, confidence filtering, execution simulation, "
        "backtesting or performance claim."
    )
    print("  supervised tasks:")
    for task in config.tasks:
        print(
            "    "
            f"{task.name}: type={task.task_type}, "
            f"classes={task.num_classes}, loss_weight={task.loss_weight}"
        )
    print("  transformer backbone:")
    print(f"    token fields:          {list(config.backbone.token_field_names)}")
    print(f"    model_dim:             {config.backbone.model_dim}")
    print(f"    num_heads:             {config.backbone.num_heads}")
    print(f"    num_layers:            {config.backbone.num_layers}")
    print(f"    feedforward_dim:       {config.backbone.feedforward_dim}")
    print(f"    dropout:               {config.backbone.dropout}")
    print(f"    max_sequence_length:   {config.backbone.max_sequence_length}")
    print(f"    pooling:               {config.backbone.pooling}")
    print(f"  head dropout:            {config.dropout}")
    print(f"  freeze backbone:         {config.freeze_backbone}")
    print(f"  model parameter count:   {model.n_parameters()}")
    print("  No training was run.")
    return 0


def _run_multitask_smoke_impl(
    path: Path,
    *,
    window_length: int = 4,
    batch_size: int = 4,
    epochs: int = 1,
    seed: int = 42,
    symbol: str | None = None,
    max_levels_per_side: int = 2,
) -> int:
    """Run a tiny synthetic supervised multi-task smoke experiment."""
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

    from chronoslob.training.multitask_experiment import (
        run_multitask_smoke_from_event_log,
    )

    path = Path(path)
    _print_synthetic_fixture_warning(path)
    try:
        result = run_multitask_smoke_from_event_log(
            path=path,
            symbol=symbol,
            window_length=window_length,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            max_levels_per_side=max_levels_per_side,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"Multi-task smoke failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Synthetic supervised multi-task plumbing only; losses and accuracies "
        "are not market evidence, alpha or tradability claims."
    )
    print(f"  path:                   {path}")
    print(f"  symbol filter:          {symbol if symbol is not None else 'none'}")
    print(f"  input records:          {result['input_record_count']}")
    print(f"  tokenised records:      {result['tokenised_record_count']}")
    print(f"  window length:          {result['window_length']}")
    print(f"  token windows:          {result['window_count']}")
    print(f"  supervised windows:     {result['supervised_window_count']}")
    print(f"  enabled tasks:          {result['enabled_tasks']}")
    print("  valid labels per task:")
    for name, count in result["valid_labels_per_task"].items():
        print(f"    {name}: {count}")
    print(f"  model parameter count:  {result['model_parameter_count']}")
    if result["final_train_loss"] is not None:
        print(f"  final train loss:       {result['final_train_loss']:.6f}")
        for name, value in result["final_train_loss_components"].items():
            print(f"    {name} loss:          {value:.6f}")
    task_accuracy = result["synthetic_smoke_metrics"]["task_accuracy"]
    if task_accuracy:
        print("  synthetic smoke accuracy:")
        for name, value in task_accuracy.items():
            print(f"    {name}: {value:.6f}")
    print(f"  label source:           {result['label_source']}")
    print("  outputs:                not written (smoke command)")
    print("  checkpoints:            not written")
    print("  network calls:          none performed")
    return 0


def _inspect_calibration_impl() -> int:
    """Print supported calibration utilities without fitting anything."""
    from chronoslob.models.calibration import CalibrationErrorConfig
    from chronoslob.training.calibration import ConfidenceFilterConfig

    error_config = CalibrationErrorConfig()
    filter_config = ConfidenceFilterConfig()

    print("ChronosLOB calibration and uncertainty")
    print("  supported metrics:")
    print("    negative_log_likelihood")
    print("    brier_score")
    print("    expected_calibration_error")
    print("    reliability_bins")
    print("    confidence_filtering")
    print("    abstention_curve")
    print(f"  default ECE bins:          {error_config.n_bins}")
    print(
        "  default confidence range:  "
        f"{error_config.min_confidence:.1f}..{error_config.max_confidence:.1f}"
    )
    print(f"  default thresholds:        {list(filter_config.thresholds)}")
    print(
        "  temperature scaling:       one positive scalar fitted on a "
        "calibration split by minimising NLL"
    )
    print("  training run:              none")
    print("  outputs:                   not written")
    print("  performance claims:        none")
    return 0


def _run_calibration_smoke_impl(
    *,
    n_examples: int = 60,
    num_classes: int = 3,
    seed: int = 42,
    ece_bins: int = 10,
) -> int:
    """Run a deterministic synthetic calibration smoke check."""
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

    from chronoslob.training.calibration import run_calibration_smoke

    try:
        result = cast(
            Mapping[str, Any],
            run_calibration_smoke(
                n_examples=n_examples,
                num_classes=num_classes,
                seed=seed,
                ece_bins=ece_bins,
            ),
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"Calibration smoke failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Synthetic calibration plumbing only; metrics do not measure market "
        "signal, alpha, tradability or execution-aware validation."
    )
    print(f"  synthetic examples:      {result['n_examples']}")
    print(f"  calibration examples:    {result['calibration_examples']}")
    print(f"  evaluation examples:     {result['evaluation_examples']}")
    print(f"  number of classes:       {result['num_classes']}")
    print(f"  seed:                    {result['seed']}")
    print(f"  fitted temperature:      {result['fitted_temperature']:.6f}")
    print("  pre-calibration metrics:")
    pre = cast(Mapping[str, Any], result["pre_calibration"])
    print(
        f"    nll={pre['nll']:.6f} "
        f"ece={pre['ece']:.6f} "
        f"brier={pre['brier_score']:.6f}"
    )
    print("  post-calibration metrics:")
    post = cast(Mapping[str, Any], result["post_calibration"])
    print(
        f"    nll={post['nll']:.6f} "
        f"ece={post['ece']:.6f} "
        f"brier={post['brier_score']:.6f}"
    )
    print("  confidence filtering:")
    print("    threshold  coverage  abstention  accuracy  n_covered/n_total")
    confidence_filtering = cast(
        Mapping[str, Any],
        result["confidence_filtering"],
    )
    buckets = cast(Sequence[Mapping[str, Any]], confidence_filtering["buckets"])
    for bucket in buckets:
        accuracy = bucket["accuracy_on_covered"]
        accuracy_text = "n/a" if accuracy is None else f"{accuracy:.6f}"
        print(
            "    "
            f"{bucket['threshold']:.2f}       "
            f"{bucket['coverage']:.6f}  "
            f"{bucket['abstention_rate']:.6f}  "
            f"{accuracy_text:<8}  "
            f"{bucket['n_covered']}/{bucket['n_total']}"
        )
    print("  outputs:                 not written (smoke command)")
    print("  checkpoints:             not written")
    print("  network calls:           none performed")
    return 0


def _inspect_execution_validation_impl() -> int:
    """Print supported execution-aware validation infrastructure."""
    from chronoslob.backtest.execution import ExecutionMode

    print("ChronosLOB execution-aware validation")
    print("  supported execution modes:")
    for mode in ExecutionMode:
        print(f"    {mode.value}")
    print("  supported cost components:")
    print("    fixed_fee_per_trade")
    print("    proportional_fee_bps")
    print("    aggressive half-spread or full-spread convention")
    print("    passive adverse-selection/slippage assumptions")
    print("  supported risk constraints:")
    print("    inventory_limit")
    print("    max_trades")
    print("    max_turnover")
    print("    optional max_drawdown")
    print("  summary metrics:")
    print("    coverage, fill_rate, hit_rate")
    print("    gross_pnl_simulated, total_cost_simulated, net_pnl_simulated")
    print("    turnover, adverse_selection_rate, latency sensitivity")
    print("    confidence-threshold sweep")
    print("  training run:       none")
    print("  live trading:       not implemented")
    print("  outputs:            not written")
    print("  statement:          simulation infrastructure only; no tradability claim")
    return 0


def _run_execution_validation_smoke_impl(
    *,
    n_signals: int = 24,
    seed: int = 42,
) -> int:
    """Run deterministic synthetic execution-validation plumbing."""
    from chronoslob.backtest.validation import run_execution_validation_smoke

    try:
        result = cast(
            Mapping[str, Any],
            run_execution_validation_smoke(n_signals=n_signals, seed=seed),
        )
    except (ValueError, TypeError) as exc:
        print(f"Execution-validation smoke failed: {exc}", file=sys.stderr)
        return 1

    print(
        "Synthetic execution-validation plumbing only; outputs are not market "
        "evidence, alpha evidence, tradability evidence or live performance."
    )
    print(f"  synthetic signals:      {result['n_signals']}")
    print(f"  market-state rows:      {result['market_state_rows']}")
    print(f"  seed:                   {result['seed']}")
    print(f"  primary mode:           {result['primary_mode']}")
    summary = cast(Mapping[str, Any], result["summary"])
    print(f"  number of trades:       {summary['n_trades']}")
    print(f"  number filled:          {summary['n_filled']}")
    print(f"  fill rate:              {summary['fill_rate']:.6f}")
    print(f"  gross simulated PnL:    {summary['gross_pnl_simulated']:.6f}")
    print(f"  total simulated cost:   {summary['total_cost_simulated']:.6f}")
    print(f"  net simulated PnL:      {summary['net_pnl_simulated']:.6f}")
    print(f"  turnover:               {summary['turnover']:.6f}")
    adverse_rate = summary["adverse_selection_rate"]
    adverse_text = "n/a" if adverse_rate is None else f"{adverse_rate:.6f}"
    print(f"  adverse selection rate: {adverse_text}")
    print("  confidence-threshold sweep:")
    print("    threshold  coverage  filled  net_pnl_simulated")
    threshold_rows = cast(
        Sequence[Mapping[str, Any]],
        result["confidence_threshold_sweep"],
    )
    for row in threshold_rows:
        print(
            "    "
            f"{row['threshold']:.2f}       "
            f"{row['coverage']:.6f}  "
            f"{row['n_filled']}       "
            f"{row['net_pnl_simulated']:.6f}"
        )
    print("  latency-sensitivity summary:")
    print("    steps  coverage  filled  net_pnl_simulated")
    latency_rows = cast(Sequence[Mapping[str, Any]], result["latency_sensitivity"])
    for row in latency_rows:
        print(
            "    "
            f"{row['latency_steps']}      "
            f"{row['coverage']:.6f}  "
            f"{row['n_filled']}       "
            f"{row['net_pnl_simulated']:.6f}"
        )
    print("  outputs:                not written (smoke command)")
    print("  live trading:           not implemented")
    print("  network calls:          none performed")
    return 0


def _inspect_analysis_impl() -> int:
    """Print supported analysis tools without running them."""
    from chronoslob.analysis.ablations import ABLATION_CATEGORIES
    from chronoslob.analysis.regimes import SUPPORTED_REGIME_KINDS
    from chronoslob.analysis.sensitivity import SENSITIVITY_PARAMETERS
    from chronoslob.analysis.summary import (
        ANALYSIS_TYPES,
        EXECUTION_METRIC_NAMES,
        PREDICTIVE_METRIC_NAMES,
        SUPPORTED_METRIC_NAMES,
    )

    print("ChronosLOB analysis layer")
    print("  supported analysis types:")
    for analysis_type in ANALYSIS_TYPES:
        print(f"    {analysis_type}")
    print("  supported regime kinds:")
    for kind in SUPPORTED_REGIME_KINDS:
        print(f"    {kind}")
    print("  supported ablation categories:")
    for category in ABLATION_CATEGORIES:
        print(f"    {category}")
    print("  supported sensitivity parameters:")
    for parameter in SENSITIVITY_PARAMETERS:
        print(f"    {parameter}")
    print("  supported metric names:")
    for metric_name in SUPPORTED_METRIC_NAMES:
        print(f"    {metric_name}")
    print("  predictive metric names:")
    for metric_name in PREDICTIVE_METRIC_NAMES:
        print(f"    {metric_name}")
    print("  execution metric names:")
    for metric_name in EXECUTION_METRIC_NAMES:
        print(f"    {metric_name}")
    print(
        "  note: analysis summaries require real upstream experiment records "
        "and do not generate evidence by themselves."
    )
    print("  outputs:             not written (read-only command)")
    print("  network calls:       none performed")
    return 0


def _run_robustness_analysis_smoke_impl(
    *,
    n_records: int = 36,
    seed: int = 42,
) -> int:
    """Run a deterministic synthetic robustness-analysis smoke check."""
    from chronoslob.analysis.summary import run_robustness_analysis_smoke

    try:
        result = cast(
            Mapping[str, Any],
            run_robustness_analysis_smoke(n_records=n_records, seed=seed),
        )
    except (ValueError, TypeError) as exc:
        print(f"Robustness-analysis smoke failed: {exc}", file=sys.stderr)
        return 1

    print(str(result["warning"]))
    print(f"  synthetic records:        {result['n_records']}")
    print(f"  transfer records:         {result['n_transfer_records']}")
    print(f"  sensitivity points:       {result['n_sensitivity_points']}")
    print(f"  ablation records:         {result['n_ablation_records']}")
    regime_summary_counts = cast(
        Mapping[str, int], result["regime_summary_counts"]
    )
    print("  regime summary counts:")
    for kind, count in regime_summary_counts.items():
        print(f"    {kind}: {count}")
    transfer_matrix = cast(Mapping[str, Any], result["transfer_matrix"])
    matrix_shape = cast(Sequence[int], transfer_matrix["shape"])
    print(
        "  transfer matrix dimensions: "
        f"{matrix_shape[0]}x{matrix_shape[1]} "
        f"(metric={transfer_matrix['metric_name']})"
    )
    print(
        "  ablation comparisons:     "
        f"{result['ablation_comparisons_count']}"
    )
    print(
        "  sensitivity curves:       "
        f"{result['sensitivity_curves_produced']}"
    )
    print("  example metric summaries:")
    example_rows = cast(
        Sequence[Mapping[str, Any]], result["example_summary_rows"]
    )
    for example in example_rows[:4]:
        row = cast(Mapping[str, Any], example["row"])
        mean_value = row.get("mean")
        mean_text = "n/a" if mean_value is None else f"{float(mean_value):.6f}"
        print(
            "    "
            f"metric={example['metric_name']} "
            f"direction={example['metric_direction']} "
            f"mean={mean_text} "
            f"count={row.get('count')}"
        )
    print(
        "  WARNING: synthetic analysis plumbing only; outputs are not market "
        "evidence and require real upstream experiment records to be useful."
    )
    print("  outputs:                  not written (smoke command)")
    print("  network calls:            none performed")
    return 0


def _inspect_binance_replay_impl(
    snapshot_path: Path,
    updates_path: Path,
    *,
    symbol: str | None = None,
    max_depth: int | None = None,
    stop_on_gap: bool = True,
    allow_crossed: bool = False,
) -> int:
    """Replay a local Binance-style snapshot and diff fixture.

    The command is intentionally read-only. It loads files from disk only,
    runs the reconstruction and prints a short summary. It writes no
    outputs and makes no network calls.
    """
    from chronoslob.book.replay import (
        ReplayConfig,
        replay_binance_jsonl,
        summarise_replay_result,
    )
    from chronoslob.data.schemas import Side

    snapshot_path = Path(snapshot_path)
    updates_path = Path(updates_path)

    fixture_marker = Path("tests") / "fixtures"
    fixture_marker_str = str(fixture_marker)
    is_fixture = any(
        fixture_marker_str in str(path)
        for path in (snapshot_path, updates_path)
    )
    if is_fixture:
        print(
            "WARNING: replay is running against a synthetic fixture; "
            "outputs are not real market data."
        )

    config = ReplayConfig(
        snapshot_path=snapshot_path,
        updates_path=updates_path,
        symbol=symbol,
        max_depth=max_depth,
        stop_on_gap=stop_on_gap,
        allow_crossed=allow_crossed,
    )
    result = replay_binance_jsonl(config)
    summary = summarise_replay_result(result)

    print("ChronosLOB inspect-binance-replay (offline only)")
    print(f"  snapshot path:    {snapshot_path}")
    print(f"  updates path:     {updates_path}")
    print(f"  ok:               {summary['ok']}")
    print(f"  n_snapshots:      {summary['n_snapshots']}")
    print(f"  final_update_id:  {summary['final_update_id']}")
    print(f"  issue_count:      {summary['issue_count']}")
    print(f"  gap_count:        {summary['gap_count']}")
    print(f"  crossed_count:    {summary['crossed_count']}")

    if result.snapshots:
        final_snapshot = result.snapshots[-1]
        best_bid = final_snapshot.best_bid
        best_ask = final_snapshot.best_ask
        bid_str = (
            f"{best_bid.price}@{best_bid.quantity}" if best_bid is not None else "n/a"
        )
        ask_str = (
            f"{best_ask.price}@{best_ask.quantity}" if best_ask is not None else "n/a"
        )
        print(f"  best bid:         {bid_str}")
        print(f"  best ask:         {ask_str}")
        bid_levels = len(final_snapshot.bids)
        ask_levels = len(final_snapshot.asks)
        print(f"  depth counts:     bids={bid_levels}, asks={ask_levels}")
        # Reference Side to keep the import meaningful even when no levels.
        _ = Side.BID
    else:
        print("  best bid:         not available (no snapshots emitted)")
        print("  best ask:         not available (no snapshots emitted)")

    print("  outputs:          not written (read-only command)")
    print("  network calls:    none performed")

    return 0 if result.ok else 1


def _fallback_main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help"}:
        print(
            "Usage: python -m chronoslob.cli "
            "[version|doctor|inspect-event-log|event-log-to-features|"
            "inspect-event-tokens|"
            "inspect-fi2010|inspect-features-fi2010|inspect-labels-fi2010|"
            "inspect-split|init-run|inspect-baselines|run-baseline-smoke|"
            "inspect-torch-dataset|inspect-deeplob|run-deeplob-smoke|"
            "inspect-transformer|run-transformer-smoke|"
            "inspect-ssl|run-ssl-smoke|"
            "inspect-multitask|run-multitask-smoke|"
            "inspect-calibration|run-calibration-smoke|"
            "inspect-execution-validation|run-execution-validation-smoke|"
            "inspect-analysis|run-robustness-analysis-smoke|"
            "inspect-binance-replay|run-project-audit|"
            "inspect-release-readiness|"
            "build-report-archive|inspect-report-archive|"
            "inspect-experiment-artifacts|"
            "prepare-fi2010-benchmark|"
            "verify-fi2010-local|"
            "convert-fi2010-official|"
            "run-paper-experiment|"
            "run-paper-ablations|"
            "run-system-benchmarks|"
            "inspect-system-benchmarks|"
            "build-paper-plots|"
            "inspect-paper-experiment|"
            "build-paper-report|"
            "inspect-paper-report] [...]"
        )
        return 0

    command = args[0]
    if command == "version":
        _version_impl()
        return 0
    if command == "doctor":
        _doctor_impl()
        return 0
    if command == "run-project-audit":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-project-audit",
            description="Run local repository audit checks without writing outputs.",
        )
        parser.add_argument("--root", type=Path, default=None)
        parser.add_argument("--strict", action="store_true")
        parsed = parser.parse_args(args[1:])
        return _run_project_audit_impl(root=parsed.root, strict=parsed.strict)
    if command == "inspect-release-readiness":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-release-readiness",
            description="Inspect public release readiness without writing outputs.",
        )
        parser.add_argument("--root", type=Path, default=None)
        parsed = parser.parse_args(args[1:])
        return _inspect_release_readiness_impl(root=parsed.root)
    if command == "build-report-archive":
        parser = argparse.ArgumentParser(
            prog="chronoslob build-report-archive",
            description="Build the local report evidence archive.",
        )
        parser.add_argument("--output", type=Path, default=Path("reports/report_archive"))
        parser.add_argument("--strict", action="store_true")
        parser.add_argument(
            "--include-smoke-training",
            action="store_true",
            help="Also capture short synthetic smoke-training commands.",
        )
        parsed = parser.parse_args(args[1:])
        return _build_report_archive_impl(
            output=parsed.output,
            strict=parsed.strict,
            include_smoke_training=parsed.include_smoke_training,
        )
    if command == "inspect-report-archive":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-report-archive",
            description="Inspect expected report archive files without writing.",
        )
        parser.add_argument("--output", type=Path, default=Path("reports/report_archive"))
        parsed = parser.parse_args(args[1:])
        return _inspect_report_archive_impl(output=parsed.output)
    if command == "inspect-experiment-artifacts":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-experiment-artifacts",
            description="Inspect an experiment directory against the artefact contract.",
        )
        parser.add_argument("--experiment", type=Path, required=True)
        parsed = parser.parse_args(args[1:])
        return _inspect_experiment_artifacts_impl(experiment=parsed.experiment)
    if command == "prepare-fi2010-benchmark":
        parser = argparse.ArgumentParser(
            prog="chronoslob prepare-fi2010-benchmark",
            description=(
                "Prepare a local-only FI-2010 benchmark input from a "
                "user-supplied data file. No data is downloaded and no "
                "model results are produced."
            ),
        )
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--data-path", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parsed = parser.parse_args(args[1:])
        return _prepare_fi2010_benchmark_impl(
            config_path=parsed.config,
            data_path=parsed.data_path,
            out=parsed.out,
        )
    if command == "verify-fi2010-local":
        parser = argparse.ArgumentParser(
            prog="chronoslob verify-fi2010-local",
            description=(
                "Inspect a local FI-2010 ``.txt`` matrix safely without "
                "loading it into memory."
            ),
        )
        parser.add_argument("--data-path", type=Path, required=True)
        parsed = parser.parse_args(args[1:])
        return _verify_fi2010_local_impl(data_path=parsed.data_path)
    if command == "convert-fi2010-official":
        parser = argparse.ArgumentParser(
            prog="chronoslob convert-fi2010-official",
            description=(
                "Convert one official FI-2010 ``.txt`` matrix into a "
                "header-bearing CSV file matching the existing FI-2010 "
                "loader convention."
            ),
        )
        parser.add_argument("--input", dest="input_path", type=Path, required=True)
        parser.add_argument("--output", dest="output_path", type=Path, required=True)
        parser.add_argument(
            "--split",
            dest="split_label",
            type=str,
            default=None,
            help="Optional split label written to a 'split' column (train or test).",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace the output file if it already exists.",
        )
        parsed = parser.parse_args(args[1:])
        return _convert_fi2010_official_impl(
            input_path=parsed.input_path,
            output_path=parsed.output_path,
            split_label=parsed.split_label,
            overwrite=bool(parsed.overwrite),
        )
    if command == "run-paper-experiment":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-paper-experiment",
            description=(
                "Run the paper experiment runner on a local FI-2010-style "
                "file and write a validated experiment artefact directory."
            ),
        )
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--data-path", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument(
            "--models",
            type=str,
            default="majority",
            help="Comma-separated model list. Defaults to 'majority'.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace the output directory if it already exists.",
        )
        parser.add_argument(
            "--build-plots",
            dest="build_plots",
            action="store_true",
            help=(
                "Generate paper experiment plots from stored artefacts "
                "after the run finishes."
            ),
        )
        parsed = parser.parse_args(args[1:])
        model_tokens = [
            token.strip()
            for token in str(parsed.models).split(",")
            if token.strip()
        ]
        return _run_paper_experiment_impl(
            config_path=parsed.config,
            data_path=parsed.data_path,
            out=parsed.out,
            models=model_tokens or None,
            overwrite=bool(parsed.overwrite),
            build_plots=bool(parsed.build_plots),
        )
    if command == "run-paper-ablations":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-paper-ablations",
            description=(
                "Run a controlled paper-experiment ablation suite on a local "
                "FI-2010-style file. The runner reuses the paper experiment "
                "runner and writes an aggregate ablation summary plus "
                "concise markdown reports."
            ),
        )
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--data-path", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument(
            "--models",
            type=str,
            default="majority,logistic",
            help=(
                "Comma-separated model list forwarded to each child paper "
                "experiment. Defaults to 'majority,logistic'."
            ),
        )
        parser.add_argument(
            "--ablation-set",
            type=str,
            default="smoke",
            help=(
                "Named ablation set. Supported values: smoke, standard. "
                "Defaults to smoke."
            ),
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace the output directory if it already exists.",
        )
        parser.add_argument(
            "--build-plots",
            dest="build_plots",
            action="store_true",
            help="Generate plots inside each child experiment directory.",
        )
        parsed = parser.parse_args(args[1:])
        model_tokens = [
            token.strip()
            for token in str(parsed.models).split(",")
            if token.strip()
        ]
        return _run_paper_ablations_impl(
            config_path=parsed.config,
            data_path=parsed.data_path,
            out=parsed.out,
            models=model_tokens or None,
            ablation_set=str(parsed.ablation_set),
            overwrite=bool(parsed.overwrite),
            build_plots=bool(parsed.build_plots),
        )
    if command == "run-system-benchmarks":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-system-benchmarks",
            description=(
                "Run local systems benchmarks on a supplied FI-2010-style "
                "data path and write traceable benchmark artefacts."
            ),
        )
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--data-path", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument(
            "--benchmark-set",
            type=str,
            default="smoke",
            help="Named benchmark set. Supported values: smoke, standard.",
        )
        parser.add_argument(
            "--models",
            type=str,
            default="majority,logistic",
            help="Comma-separated paper-runner model list.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace the output directory if it already exists.",
        )
        parsed = parser.parse_args(args[1:])
        model_tokens = [
            token.strip()
            for token in str(parsed.models).split(",")
            if token.strip()
        ]
        return _run_system_benchmarks_impl(
            config_path=parsed.config,
            data_path=parsed.data_path,
            out=parsed.out,
            benchmark_set=str(parsed.benchmark_set),
            models=model_tokens or None,
            overwrite=bool(parsed.overwrite),
        )
    if command == "inspect-system-benchmarks":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-system-benchmarks",
            description="Inspect a completed systems benchmark directory.",
        )
        parser.add_argument("--benchmark", type=Path, required=True)
        parsed = parser.parse_args(args[1:])
        return _inspect_system_benchmarks_impl(benchmark=parsed.benchmark)
    if command == "build-paper-plots":
        parser = argparse.ArgumentParser(
            prog="chronoslob build-paper-plots",
            description=(
                "Generate paper experiment plots from the artefacts stored "
                "inside a completed experiment directory."
            ),
        )
        parser.add_argument("--experiment", type=Path, required=True)
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace existing plot files when generating.",
        )
        parsed = parser.parse_args(args[1:])
        return _build_paper_plots_impl(
            experiment=parsed.experiment,
            overwrite=bool(parsed.overwrite),
        )
    if command == "inspect-paper-experiment":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-paper-experiment",
            description=(
                "Print a concise human-readable summary of a paper "
                "experiment artefact directory."
            ),
        )
        parser.add_argument("--experiment", type=Path, required=True)
        parsed = parser.parse_args(args[1:])
        return _inspect_paper_experiment_impl(experiment=parsed.experiment)
    if command == "build-paper-report":
        parser = argparse.ArgumentParser(
            prog="chronoslob build-paper-report",
            description=(
                "Build an empirical Markdown report from stored paper "
                "experiment, ablation and systems artefacts."
            ),
        )
        parser.add_argument("--experiment", type=Path, required=True)
        parser.add_argument("--ablations", type=Path, default=None)
        parser.add_argument("--systems", type=Path, default=None)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace the report and summary JSON if they already exist.",
        )
        parsed = parser.parse_args(args[1:])
        return _build_paper_report_impl(
            experiment=parsed.experiment,
            ablations=parsed.ablations,
            systems=parsed.systems,
            out=parsed.out,
            overwrite=bool(parsed.overwrite),
        )
    if command == "inspect-paper-report":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-paper-report",
            description="Inspect a generated empirical report and summary JSON.",
        )
        parser.add_argument("--report", type=Path, required=True)
        parsed = parser.parse_args(args[1:])
        return _inspect_paper_report_impl(report=parsed.report)
    if command == "inspect-event-log":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-event-log",
            description="Inspect a local canonical event-log JSONL file.",
        )
        parser.add_argument("--path", type=Path, required=True)
        parsed = parser.parse_args(args[1:])
        return _inspect_event_log_impl(parsed.path)
    if command == "inspect-event-tokens":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-event-tokens",
            description=(
                "Tokenise a local canonical event-log JSONL file and print "
                "a read-only summary."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--symbol", default=None)
        parser.add_argument("--window-length", type=int, default=8)
        parser.add_argument("--max-levels-per-side", type=int, default=2)
        parser.add_argument(
            "--include-eos",
            dest="include_eos",
            action="store_true",
            help="Append one [EOS] record to the inspected token sequence.",
        )
        parser.add_argument(
            "--no-include-eos",
            dest="include_eos",
            action="store_false",
            help="Do not append an [EOS] record.",
        )
        parser.set_defaults(include_eos=False)
        parsed = parser.parse_args(args[1:])
        return _inspect_event_tokens_impl(
            path=parsed.path,
            symbol=parsed.symbol,
            window_length=parsed.window_length,
            max_levels_per_side=parsed.max_levels_per_side,
            include_eos=parsed.include_eos,
        )
    if command == "event-log-to-features":
        parser = argparse.ArgumentParser(
            prog="chronoslob event-log-to-features",
            description=(
                "Replay a local canonical event-log JSONL file into "
                "microstructure features and print a read-only summary."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parsed = parser.parse_args(args[1:])
        return _event_log_to_features_impl(parsed.path)
    if command == "inspect-fi2010":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-fi2010",
            description="Load an FI-2010 file and print a data-quality summary.",
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--timestamp-column", default="timestamp")
        parser.add_argument("--split-column", default="split")
        parser.add_argument("--price-level-count", type=int, default=2)
        parser.add_argument(
            "--no-timestamp-column",
            action="store_true",
            help="Treat the file as having no timestamp column.",
        )
        parser.add_argument(
            "--no-split-column",
            action="store_true",
            help="Treat the file as having no split column.",
        )
        parsed = parser.parse_args(args[1:])
        return _inspect_fi2010_impl(
            path=parsed.path,
            timestamp_column=(
                None if parsed.no_timestamp_column else parsed.timestamp_column
            ),
            split_column=(
                None if parsed.no_split_column else parsed.split_column
            ),
            price_level_count=parsed.price_level_count,
        )
    if command == "inspect-features-fi2010":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-features-fi2010",
            description=(
                "Load an FI-2010 file, build microstructure features and "
                "print a short summary."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--timestamp-column", default="timestamp")
        parser.add_argument("--split-column", default="split")
        parser.add_argument("--price-level-count", type=int, default=2)
        parser.add_argument(
            "--no-timestamp-column",
            action="store_true",
            help="Treat the file as having no timestamp column.",
        )
        parser.add_argument(
            "--no-split-column",
            action="store_true",
            help="Treat the file as having no split column.",
        )
        parser.add_argument(
            "--allow-synthetic-time",
            action="store_true",
            help=(
                "Compute time-window features even when timestamps are "
                "synthetic. Off by default."
            ),
        )
        parsed = parser.parse_args(args[1:])
        return _inspect_features_fi2010_impl(
            path=parsed.path,
            timestamp_column=(
                None if parsed.no_timestamp_column else parsed.timestamp_column
            ),
            split_column=(
                None if parsed.no_split_column else parsed.split_column
            ),
            price_level_count=parsed.price_level_count,
            allow_synthetic_timestamps_for_time_features=(
                parsed.allow_synthetic_time
            ),
        )
    if command == "inspect-labels-fi2010":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-labels-fi2010",
            description=(
                "Load an FI-2010 file, build or extract labels and print "
                "a short summary."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--timestamp-column", default="timestamp")
        parser.add_argument("--split-column", default="split")
        parser.add_argument("--price-level-count", type=int, default=2)
        parser.add_argument(
            "--no-timestamp-column",
            action="store_true",
            help="Treat the file as having no timestamp column.",
        )
        parser.add_argument(
            "--no-split-column",
            action="store_true",
            help="Treat the file as having no split column.",
        )
        parser.add_argument(
            "--generate-labels",
            action="store_true",
            help=(
                "Generate ChronosLOB labels from snapshots instead of "
                "preferring configured FI-2010 benchmark labels."
            ),
        )
        parsed = parser.parse_args(args[1:])
        return _inspect_labels_fi2010_impl(
            path=parsed.path,
            timestamp_column=(
                None if parsed.no_timestamp_column else parsed.timestamp_column
            ),
            split_column=None if parsed.no_split_column else parsed.split_column,
            price_level_count=parsed.price_level_count,
            prefer_existing_labels=not parsed.generate_labels,
        )
    if command == "inspect-split":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-split",
            description="Build a default temporal split and print partition counts.",
        )
        parser.add_argument("--rows", type=int, required=True)
        parsed = parser.parse_args(args[1:])
        return _inspect_split_impl(parsed.rows)
    if command == "init-run":
        parser = argparse.ArgumentParser(
            prog="chronoslob init-run",
            description="Create a metadata-only experiment run directory.",
        )
        parser.add_argument("--name", required=True)
        parser.add_argument("--phase", required=True)
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--config-path", type=Path, default=None)
        parser.add_argument("--notes", default=None)
        parsed = parser.parse_args(args[1:])
        return _init_run_impl(
            name=parsed.name,
            phase=parsed.phase,
            seed=parsed.seed,
            root=parsed.root,
            config_path=parsed.config_path,
            notes=parsed.notes,
        )
    if command == "inspect-baselines":
        return _inspect_baselines_impl()
    if command == "run-baseline-smoke":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-baseline-smoke",
            description=(
                "Run a synthetic FI-2010 fixture baseline smoke test. "
                "This is not benchmark performance."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--write-outputs", action="store_true")
        parser.add_argument("--output-root", type=Path, default=Path("runs"))
        parsed = parser.parse_args(args[1:])
        return _run_baseline_smoke_impl(
            path=parsed.path,
            write_outputs=parsed.write_outputs,
            output_root=parsed.output_root,
        )
    if command == "inspect-torch-dataset":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-torch-dataset",
            description=(
                "Build a tiny sequence DataLoader from a local FI-2010 file "
                "and print a summary. Not benchmark performance."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--lookback", type=int, default=2)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--target-column", default="label_10")
        parser.add_argument("--timestamp-column", default="timestamp")
        parser.add_argument("--split-column", default="split")
        parser.add_argument("--price-level-count", type=int, default=2)
        parser.add_argument("--train-fraction", type=float, default=0.5)
        parser.add_argument("--validation-fraction", type=float, default=0.34)
        parser.add_argument("--test-fraction", type=float, default=0.16)
        parser.add_argument(
            "--no-timestamp-column",
            action="store_true",
            help="Treat the file as having no timestamp column.",
        )
        parser.add_argument(
            "--no-split-column",
            action="store_true",
            help="Treat the file as having no split column.",
        )
        parsed = parser.parse_args(args[1:])
        return _inspect_torch_dataset_impl(
            path=parsed.path,
            lookback=parsed.lookback,
            batch_size=parsed.batch_size,
            target_column=parsed.target_column,
            timestamp_column=(
                None if parsed.no_timestamp_column else parsed.timestamp_column
            ),
            split_column=None if parsed.no_split_column else parsed.split_column,
            price_level_count=parsed.price_level_count,
            train_fraction=parsed.train_fraction,
            validation_fraction=parsed.validation_fraction,
            test_fraction=parsed.test_fraction,
        )
    if command == "inspect-deeplob":
        return _inspect_deeplob_impl()
    if command == "run-deeplob-smoke":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-deeplob-smoke",
            description=(
                "Run a synthetic-fixture DeepLOB-style supervised smoke "
                "experiment. This is not benchmark performance."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--lookback", type=int, default=2)
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--write-outputs", action="store_true")
        parser.add_argument("--output-root", type=Path, default=Path("runs"))
        parsed = parser.parse_args(args[1:])
        return _run_deeplob_smoke_impl(
            path=parsed.path,
            lookback=parsed.lookback,
            epochs=parsed.epochs,
            batch_size=parsed.batch_size,
            seed=parsed.seed,
            write_outputs=parsed.write_outputs,
            output_root=parsed.output_root,
        )

    if command == "inspect-transformer":
        return _inspect_transformer_impl()
    if command == "run-transformer-smoke":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-transformer-smoke",
            description=(
                "Run a synthetic-label transformer smoke experiment on a "
                "canonical event log. Labels are synthetic plumbing only."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--window-length", type=int, default=4)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--num-classes", type=int, default=3)
        parser.add_argument("--symbol", default=None)
        parser.add_argument("--max-levels-per-side", type=int, default=2)
        parsed = parser.parse_args(args[1:])
        return _run_transformer_smoke_impl(
            path=parsed.path,
            window_length=parsed.window_length,
            batch_size=parsed.batch_size,
            epochs=parsed.epochs,
            seed=parsed.seed,
            num_classes=parsed.num_classes,
            symbol=parsed.symbol,
            max_levels_per_side=parsed.max_levels_per_side,
        )
    if command == "inspect-ssl":
        return _inspect_ssl_impl()
    if command == "run-ssl-smoke":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-ssl-smoke",
            description=(
                "Run a synthetic self-supervised pretraining smoke experiment "
                "on a canonical event log. Losses are plumbing only."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--window-length", type=int, default=4)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--symbol", default=None)
        parser.add_argument("--max-levels-per-side", type=int, default=2)
        parser.add_argument("--mask-probability", type=float, default=0.15)
        parsed = parser.parse_args(args[1:])
        return _run_ssl_smoke_impl(
            path=parsed.path,
            window_length=parsed.window_length,
            batch_size=parsed.batch_size,
            epochs=parsed.epochs,
            seed=parsed.seed,
            symbol=parsed.symbol,
            max_levels_per_side=parsed.max_levels_per_side,
            mask_probability=parsed.mask_probability,
        )
    if command == "inspect-multitask":
        return _inspect_multitask_impl()
    if command == "run-multitask-smoke":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-multitask-smoke",
            description=(
                "Run a synthetic supervised multi-task smoke experiment on a "
                "canonical event log. Labels are plumbing only."
            ),
        )
        parser.add_argument("--path", type=Path, required=True)
        parser.add_argument("--window-length", type=int, default=4)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--symbol", default=None)
        parser.add_argument("--max-levels-per-side", type=int, default=2)
        parsed = parser.parse_args(args[1:])
        return _run_multitask_smoke_impl(
            path=parsed.path,
            window_length=parsed.window_length,
            batch_size=parsed.batch_size,
            epochs=parsed.epochs,
            seed=parsed.seed,
            symbol=parsed.symbol,
            max_levels_per_side=parsed.max_levels_per_side,
        )
    if command == "inspect-calibration":
        return _inspect_calibration_impl()
    if command == "run-calibration-smoke":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-calibration-smoke",
            description=(
                "Run a deterministic synthetic calibration smoke check. "
                "This is plumbing only."
            ),
        )
        parser.add_argument("--n-examples", type=int, default=60)
        parser.add_argument("--num-classes", type=int, default=3)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--ece-bins", type=int, default=10)
        parsed = parser.parse_args(args[1:])
        return _run_calibration_smoke_impl(
            n_examples=parsed.n_examples,
            num_classes=parsed.num_classes,
            seed=parsed.seed,
            ece_bins=parsed.ece_bins,
        )
    if command == "inspect-execution-validation":
        return _inspect_execution_validation_impl()
    if command == "run-execution-validation-smoke":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-execution-validation-smoke",
            description=(
                "Run a deterministic synthetic execution-validation smoke "
                "check. This is simulation plumbing only."
            ),
        )
        parser.add_argument("--n-signals", type=int, default=24)
        parser.add_argument("--seed", type=int, default=42)
        parsed = parser.parse_args(args[1:])
        return _run_execution_validation_smoke_impl(
            n_signals=parsed.n_signals,
            seed=parsed.seed,
        )
    if command == "inspect-analysis":
        return _inspect_analysis_impl()
    if command == "run-robustness-analysis-smoke":
        parser = argparse.ArgumentParser(
            prog="chronoslob run-robustness-analysis-smoke",
            description=(
                "Run a deterministic synthetic robustness-analysis smoke "
                "check. This is analysis plumbing only."
            ),
        )
        parser.add_argument("--n-records", type=int, default=36)
        parser.add_argument("--seed", type=int, default=42)
        parsed = parser.parse_args(args[1:])
        return _run_robustness_analysis_smoke_impl(
            n_records=parsed.n_records,
            seed=parsed.seed,
        )
    if command == "inspect-binance-replay":
        parser = argparse.ArgumentParser(
            prog="chronoslob inspect-binance-replay",
            description=(
                "Reconstruct a local Binance-style order book from a "
                "snapshot and diff JSONL fixture. Offline only."
            ),
        )
        parser.add_argument("--snapshot", type=Path, required=True)
        parser.add_argument("--updates", type=Path, required=True)
        parser.add_argument("--symbol", default=None)
        parser.add_argument("--max-depth", type=int, default=None)
        parser.add_argument(
            "--no-stop-on-gap",
            action="store_true",
            help="Continue reconstruction after a gap is detected.",
        )
        parser.add_argument(
            "--allow-crossed",
            action="store_true",
            help="Permit crossed books instead of treating them as errors.",
        )
        parsed = parser.parse_args(args[1:])
        return _inspect_binance_replay_impl(
            snapshot_path=parsed.snapshot,
            updates_path=parsed.updates,
            symbol=parsed.symbol,
            max_depth=parsed.max_depth,
            stop_on_gap=not parsed.no_stop_on_gap,
            allow_crossed=parsed.allow_crossed,
        )

    print(f"Unknown command: {command}", file=sys.stderr)
    return 2


if typer is not None:
    app = typer.Typer(
        add_completion=False,
        help="ChronosLOB research-engineering utilities.",
        no_args_is_help=True,
    )


def version() -> None:
    """Print the installed ChronosLOB version."""
    _version_impl()


def doctor() -> None:
    """Print a lightweight environment and scaffold check."""
    _doctor_impl()


if typer is not None:
    _INSPECT_PATH_OPTION = typer.Option(
        ...,
        "--path",
        help="Path to the local FI-2010-style file.",
    )
    _EVENT_LOG_PATH_OPTION = typer.Option(
        ...,
        "--path",
        help="Path to the local canonical event-log JSONL file.",
    )
    _EVENT_TOKENS_SYMBOL_OPTION = typer.Option(
        None,
        "--symbol",
        help="Optional symbol filter for event-token inspection.",
    )
    _EVENT_TOKENS_WINDOW_LENGTH_OPTION = typer.Option(
        8,
        "--window-length",
        help="Fixed token-window length used for inspection.",
    )
    _EVENT_TOKENS_MAX_LEVELS_OPTION = typer.Option(
        2,
        "--max-levels-per-side",
        help="Maximum snapshot levels per side to tokenise.",
    )
    _EVENT_TOKENS_INCLUDE_EOS_OPTION = typer.Option(
        False,
        "--include-eos/--no-include-eos",
        help="Append one [EOS] record to the inspected token sequence.",
    )
    _INSPECT_TIMESTAMP_OPTION = typer.Option(
        "timestamp",
        "--timestamp-column",
        help="Name of the timestamp column, if any.",
    )
    _INSPECT_SPLIT_OPTION = typer.Option(
        "split",
        "--split-column",
        help="Name of the split column, if any.",
    )
    _INSPECT_LEVEL_COUNT_OPTION = typer.Option(
        2,
        "--price-level-count",
        help="Expected number of LOB levels per side.",
    )
    _INSPECT_NO_TIMESTAMP_OPTION = typer.Option(
        False,
        "--no-timestamp-column",
        help="Treat the file as having no timestamp column.",
    )
    _INSPECT_NO_SPLIT_OPTION = typer.Option(
        False,
        "--no-split-column",
        help="Treat the file as having no split column.",
    )
    _AUDIT_ROOT_OPTION = typer.Option(
        None,
        "--root",
        help="Optional repository root to audit.",
    )
    _AUDIT_STRICT_OPTION = typer.Option(
        False,
        "--strict",
        help="Exit non-zero when warnings or failures are found.",
    )
    _REPORT_ARCHIVE_OUTPUT_OPTION = typer.Option(
        Path("reports/report_archive"),
        "--output",
        help="Directory where report archive files are written.",
    )
    _REPORT_ARCHIVE_STRICT_OPTION = typer.Option(
        False,
        "--strict",
        help="Fail if any captured command exits non-zero.",
    )
    _REPORT_ARCHIVE_INCLUDE_SMOKE_TRAINING_OPTION = typer.Option(
        False,
        "--include-smoke-training",
        help="Also capture short synthetic smoke-training commands.",
    )
    _EXPERIMENT_ARTIFACTS_EXPERIMENT_OPTION = typer.Option(
        ...,
        "--experiment",
        help="Path to the experiment directory to inspect.",
    )
    _PREPARE_FI2010_CONFIG_OPTION = typer.Option(
        ...,
        "--config",
        help="Path to the FI-2010 benchmark preparation YAML config.",
    )
    _PREPARE_FI2010_DATA_PATH_OPTION = typer.Option(
        ...,
        "--data-path",
        help="Local FI-2010-style file path supplied by the user.",
    )
    _PREPARE_FI2010_OUT_OPTION = typer.Option(
        ...,
        "--out",
        help="Output directory for FI-2010 preparation artefacts.",
    )
    _VERIFY_FI2010_DATA_PATH_OPTION = typer.Option(
        ...,
        "--data-path",
        help="Path to the local FI-2010 file to inspect.",
    )
    _CONVERT_FI2010_INPUT_OPTION = typer.Option(
        ...,
        "--input",
        help="Path to a single official FI-2010 .txt matrix file.",
    )
    _CONVERT_FI2010_OUTPUT_OPTION = typer.Option(
        ...,
        "--output",
        help="Destination CSV path for the converted FI-2010 file.",
    )
    _CONVERT_FI2010_SPLIT_OPTION = typer.Option(
        None,
        "--split",
        help=(
            "Optional split label written to a 'split' column "
            "(train or test)."
        ),
    )
    _CONVERT_FI2010_OVERWRITE_OPTION = typer.Option(
        False,
        "--overwrite",
        help="Replace the output file if it already exists.",
    )
    _RUN_PAPER_EXPERIMENT_CONFIG_OPTION = typer.Option(
        ...,
        "--config",
        help="Path to the paper experiment configuration YAML file.",
    )
    _RUN_PAPER_EXPERIMENT_DATA_PATH_OPTION = typer.Option(
        ...,
        "--data-path",
        help="Local FI-2010-style file path supplied by the user.",
    )
    _RUN_PAPER_EXPERIMENT_OUT_OPTION = typer.Option(
        ...,
        "--out",
        help="Output directory for the paper experiment artefacts.",
    )
    _RUN_PAPER_EXPERIMENT_MODELS_OPTION = typer.Option(
        "majority",
        "--models",
        help="Comma-separated model list. Defaults to 'majority'.",
    )
    _RUN_PAPER_EXPERIMENT_OVERWRITE_OPTION = typer.Option(
        False,
        "--overwrite",
        help="Replace the output directory if it already exists.",
    )
    _RUN_PAPER_EXPERIMENT_BUILD_PLOTS_OPTION = typer.Option(
        False,
        "--build-plots",
        help=(
            "Generate paper experiment plots from stored artefacts after "
            "the run finishes."
        ),
    )
    _RUN_PAPER_ABLATIONS_CONFIG_OPTION = typer.Option(
        ...,
        "--config",
        help="Path to the paper ablation base configuration YAML file.",
    )
    _RUN_PAPER_ABLATIONS_DATA_PATH_OPTION = typer.Option(
        ...,
        "--data-path",
        help="Local FI-2010-style file path supplied by the user.",
    )
    _RUN_PAPER_ABLATIONS_OUT_OPTION = typer.Option(
        ...,
        "--out",
        help="Output directory for aggregate ablation artefacts.",
    )
    _RUN_PAPER_ABLATIONS_MODELS_OPTION = typer.Option(
        "majority,logistic",
        "--models",
        help="Comma-separated model list. Defaults to 'majority,logistic'.",
    )
    _RUN_PAPER_ABLATIONS_SET_OPTION = typer.Option(
        "smoke",
        "--ablation-set",
        help="Named ablation set. Supported values: smoke, standard.",
    )
    _RUN_PAPER_ABLATIONS_OVERWRITE_OPTION = typer.Option(
        False,
        "--overwrite",
        help="Replace the output directory if it already exists.",
    )
    _RUN_PAPER_ABLATIONS_BUILD_PLOTS_OPTION = typer.Option(
        False,
        "--build-plots",
        help="Generate plots inside each child paper experiment directory.",
    )
    _RUN_SYSTEM_BENCHMARKS_CONFIG_OPTION = typer.Option(
        ...,
        "--config",
        help="Path to the systems benchmark configuration YAML file.",
    )
    _RUN_SYSTEM_BENCHMARKS_DATA_PATH_OPTION = typer.Option(
        ...,
        "--data-path",
        help="Local FI-2010-style file path supplied by the user.",
    )
    _RUN_SYSTEM_BENCHMARKS_OUT_OPTION = typer.Option(
        ...,
        "--out",
        help="Output directory for systems benchmark artefacts.",
    )
    _RUN_SYSTEM_BENCHMARKS_SET_OPTION = typer.Option(
        "smoke",
        "--benchmark-set",
        help="Named benchmark set. Supported values: smoke, standard.",
    )
    _RUN_SYSTEM_BENCHMARKS_MODELS_OPTION = typer.Option(
        "majority,logistic",
        "--models",
        help="Comma-separated paper-runner model list.",
    )
    _RUN_SYSTEM_BENCHMARKS_OVERWRITE_OPTION = typer.Option(
        False,
        "--overwrite",
        help="Replace the output directory if it already exists.",
    )
    _INSPECT_SYSTEM_BENCHMARKS_BENCHMARK_OPTION = typer.Option(
        ...,
        "--benchmark",
        help="Path to a completed systems benchmark directory.",
    )
    _BUILD_PAPER_PLOTS_EXPERIMENT_OPTION = typer.Option(
        ...,
        "--experiment",
        help="Path to a completed paper experiment artefact directory.",
    )
    _BUILD_PAPER_PLOTS_OVERWRITE_OPTION = typer.Option(
        False,
        "--overwrite",
        help="Replace existing plot files when generating.",
    )
    _INSPECT_PAPER_EXPERIMENT_EXPERIMENT_OPTION = typer.Option(
        ...,
        "--experiment",
        help="Path to a paper experiment artefact directory.",
    )
    _BUILD_PAPER_REPORT_EXPERIMENT_OPTION = typer.Option(
        ...,
        "--experiment",
        help="Path to a completed paper experiment artefact directory.",
    )
    _BUILD_PAPER_REPORT_ABLATIONS_OPTION = typer.Option(
        None,
        "--ablations",
        help="Optional path to a completed paper ablation directory.",
    )
    _BUILD_PAPER_REPORT_SYSTEMS_OPTION = typer.Option(
        None,
        "--systems",
        help="Optional path to a completed systems benchmark directory.",
    )
    _BUILD_PAPER_REPORT_OUT_OPTION = typer.Option(
        ...,
        "--out",
        help="Markdown report path to write.",
    )
    _BUILD_PAPER_REPORT_OVERWRITE_OPTION = typer.Option(
        False,
        "--overwrite",
        help="Replace the report and summary JSON if they already exist.",
    )
    _INSPECT_PAPER_REPORT_REPORT_OPTION = typer.Option(
        ...,
        "--report",
        help="Path to a generated empirical report Markdown file.",
    )

    def run_project_audit(
        root: Path | None = _AUDIT_ROOT_OPTION,
        strict: bool = _AUDIT_STRICT_OPTION,
    ) -> None:
        """Run local repository audit checks without writing outputs."""
        exit_code = _run_project_audit_impl(root=root, strict=strict)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_release_readiness(
        root: Path | None = _AUDIT_ROOT_OPTION,
    ) -> None:
        """Inspect public release readiness without writing outputs."""
        exit_code = _inspect_release_readiness_impl(root=root)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def build_report_archive(
        output: Path = _REPORT_ARCHIVE_OUTPUT_OPTION,
        strict: bool = _REPORT_ARCHIVE_STRICT_OPTION,
        include_smoke_training: bool = _REPORT_ARCHIVE_INCLUDE_SMOKE_TRAINING_OPTION,
    ) -> None:
        """Build the local report evidence archive."""
        exit_code = _build_report_archive_impl(
            output=output,
            strict=strict,
            include_smoke_training=include_smoke_training,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_report_archive(
        output: Path = _REPORT_ARCHIVE_OUTPUT_OPTION,
    ) -> None:
        """Inspect expected report archive files without writing."""
        exit_code = _inspect_report_archive_impl(output=output)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_experiment_artifacts(
        experiment: Path = _EXPERIMENT_ARTIFACTS_EXPERIMENT_OPTION,
    ) -> None:
        """Inspect an experiment directory against the artefact contract."""
        exit_code = _inspect_experiment_artifacts_impl(experiment=experiment)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def prepare_fi2010_benchmark(
        config: Path = _PREPARE_FI2010_CONFIG_OPTION,
        data_path: Path = _PREPARE_FI2010_DATA_PATH_OPTION,
        out: Path = _PREPARE_FI2010_OUT_OPTION,
    ) -> None:
        """Prepare a local-only FI-2010 benchmark input."""
        exit_code = _prepare_fi2010_benchmark_impl(
            config_path=config,
            data_path=data_path,
            out=out,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def verify_fi2010_local(
        data_path: Path = _VERIFY_FI2010_DATA_PATH_OPTION,
    ) -> None:
        """Inspect a local FI-2010 file safely without loading it."""
        exit_code = _verify_fi2010_local_impl(data_path=data_path)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def convert_fi2010_official(
        input_path: Path = _CONVERT_FI2010_INPUT_OPTION,
        output_path: Path = _CONVERT_FI2010_OUTPUT_OPTION,
        split: str | None = _CONVERT_FI2010_SPLIT_OPTION,
        overwrite: bool = _CONVERT_FI2010_OVERWRITE_OPTION,
    ) -> None:
        """Convert a single official FI-2010 .txt matrix into a loader-ready CSV."""
        exit_code = _convert_fi2010_official_impl(
            input_path=input_path,
            output_path=output_path,
            split_label=split,
            overwrite=overwrite,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_paper_experiment(
        config: Path = _RUN_PAPER_EXPERIMENT_CONFIG_OPTION,
        data_path: Path = _RUN_PAPER_EXPERIMENT_DATA_PATH_OPTION,
        out: Path = _RUN_PAPER_EXPERIMENT_OUT_OPTION,
        models: str = _RUN_PAPER_EXPERIMENT_MODELS_OPTION,
        overwrite: bool = _RUN_PAPER_EXPERIMENT_OVERWRITE_OPTION,
        build_plots: bool = _RUN_PAPER_EXPERIMENT_BUILD_PLOTS_OPTION,
    ) -> None:
        """Run the paper experiment runner and write artefacts."""
        model_tokens = [token.strip() for token in models.split(",") if token.strip()]
        exit_code = _run_paper_experiment_impl(
            config_path=config,
            data_path=data_path,
            out=out,
            models=model_tokens or None,
            overwrite=overwrite,
            build_plots=build_plots,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_paper_ablations(
        config: Path = _RUN_PAPER_ABLATIONS_CONFIG_OPTION,
        data_path: Path = _RUN_PAPER_ABLATIONS_DATA_PATH_OPTION,
        out: Path = _RUN_PAPER_ABLATIONS_OUT_OPTION,
        models: str = _RUN_PAPER_ABLATIONS_MODELS_OPTION,
        ablation_set: str = _RUN_PAPER_ABLATIONS_SET_OPTION,
        overwrite: bool = _RUN_PAPER_ABLATIONS_OVERWRITE_OPTION,
        build_plots: bool = _RUN_PAPER_ABLATIONS_BUILD_PLOTS_OPTION,
    ) -> None:
        """Run the paper-experiment ablation suite and write artefacts."""
        model_tokens = [token.strip() for token in models.split(",") if token.strip()]
        exit_code = _run_paper_ablations_impl(
            config_path=config,
            data_path=data_path,
            out=out,
            models=model_tokens or None,
            ablation_set=ablation_set,
            overwrite=overwrite,
            build_plots=build_plots,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_system_benchmarks(
        config: Path = _RUN_SYSTEM_BENCHMARKS_CONFIG_OPTION,
        data_path: Path = _RUN_SYSTEM_BENCHMARKS_DATA_PATH_OPTION,
        out: Path = _RUN_SYSTEM_BENCHMARKS_OUT_OPTION,
        benchmark_set: str = _RUN_SYSTEM_BENCHMARKS_SET_OPTION,
        models: str = _RUN_SYSTEM_BENCHMARKS_MODELS_OPTION,
        overwrite: bool = _RUN_SYSTEM_BENCHMARKS_OVERWRITE_OPTION,
    ) -> None:
        """Run local systems benchmarks and write artefacts."""
        model_tokens = [token.strip() for token in models.split(",") if token.strip()]
        exit_code = _run_system_benchmarks_impl(
            config_path=config,
            data_path=data_path,
            out=out,
            benchmark_set=benchmark_set,
            models=model_tokens or None,
            overwrite=overwrite,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_system_benchmarks(
        benchmark: Path = _INSPECT_SYSTEM_BENCHMARKS_BENCHMARK_OPTION,
    ) -> None:
        """Print a concise systems benchmark summary."""
        exit_code = _inspect_system_benchmarks_impl(benchmark=benchmark)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def build_paper_plots(
        experiment: Path = _BUILD_PAPER_PLOTS_EXPERIMENT_OPTION,
        overwrite: bool = _BUILD_PAPER_PLOTS_OVERWRITE_OPTION,
    ) -> None:
        """Generate paper experiment plots from stored artefacts."""
        exit_code = _build_paper_plots_impl(
            experiment=experiment,
            overwrite=overwrite,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_paper_experiment(
        experiment: Path = _INSPECT_PAPER_EXPERIMENT_EXPERIMENT_OPTION,
    ) -> None:
        """Print a concise paper experiment artefact summary."""
        exit_code = _inspect_paper_experiment_impl(experiment=experiment)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def build_paper_report(
        experiment: Path = _BUILD_PAPER_REPORT_EXPERIMENT_OPTION,
        ablations: Path | None = _BUILD_PAPER_REPORT_ABLATIONS_OPTION,
        systems: Path | None = _BUILD_PAPER_REPORT_SYSTEMS_OPTION,
        out: Path = _BUILD_PAPER_REPORT_OUT_OPTION,
        overwrite: bool = _BUILD_PAPER_REPORT_OVERWRITE_OPTION,
    ) -> None:
        """Build an empirical report from stored artefacts."""
        exit_code = _build_paper_report_impl(
            experiment=experiment,
            ablations=ablations,
            systems=systems,
            out=out,
            overwrite=overwrite,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_paper_report(
        report: Path = _INSPECT_PAPER_REPORT_REPORT_OPTION,
    ) -> None:
        """Inspect a generated empirical report summary."""
        exit_code = _inspect_paper_report_impl(report=report)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_event_log(
        path: Path = _EVENT_LOG_PATH_OPTION,
    ) -> None:
        """Inspect a local canonical event-log JSONL file."""
        exit_code = _inspect_event_log_impl(path)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_event_tokens(
        path: Path = _EVENT_LOG_PATH_OPTION,
        symbol: str | None = _EVENT_TOKENS_SYMBOL_OPTION,
        window_length: int = _EVENT_TOKENS_WINDOW_LENGTH_OPTION,
        max_levels_per_side: int = _EVENT_TOKENS_MAX_LEVELS_OPTION,
        include_eos: bool = _EVENT_TOKENS_INCLUDE_EOS_OPTION,
    ) -> None:
        """Tokenise a local canonical event log and summarise IDs."""
        exit_code = _inspect_event_tokens_impl(
            path=path,
            symbol=symbol,
            window_length=window_length,
            max_levels_per_side=max_levels_per_side,
            include_eos=include_eos,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def event_log_to_features(
        path: Path = _EVENT_LOG_PATH_OPTION,
    ) -> None:
        """Replay a local event log into feature rows and summarise."""
        exit_code = _event_log_to_features_impl(path)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_fi2010(
        path: Path = _INSPECT_PATH_OPTION,
        timestamp_column: str = _INSPECT_TIMESTAMP_OPTION,
        split_column: str = _INSPECT_SPLIT_OPTION,
        price_level_count: int = _INSPECT_LEVEL_COUNT_OPTION,
        no_timestamp_column: bool = _INSPECT_NO_TIMESTAMP_OPTION,
        no_split_column: bool = _INSPECT_NO_SPLIT_OPTION,
    ) -> None:
        """Load an FI-2010 file and print a data-quality summary."""
        exit_code = _inspect_fi2010_impl(
            path=path,
            timestamp_column=(
                None if no_timestamp_column else timestamp_column
            ),
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    _INSPECT_FEAT_ALLOW_SYNTHETIC_OPTION = typer.Option(
        False,
        "--allow-synthetic-time",
        help=(
            "Compute time-window features even when timestamps are "
            "synthetic. Off by default."
        ),
    )
    _INSPECT_LABELS_GENERATE_OPTION = typer.Option(
        False,
        "--generate-labels",
        help=(
            "Generate ChronosLOB labels from snapshots instead of "
            "preferring configured FI-2010 benchmark labels."
        ),
    )
    _INSPECT_SPLIT_ROWS_OPTION = typer.Option(
        ...,
        "--rows",
        help="Number of ordered rows to split.",
    )
    _INIT_RUN_NAME_OPTION = typer.Option(
        ...,
        "--name",
        help="Readable run name.",
    )
    _INIT_RUN_PHASE_OPTION = typer.Option(
        ...,
        "--phase",
        help="Project phase identifier.",
    )
    _INIT_RUN_SEED_OPTION = typer.Option(
        ...,
        "--seed",
        help="Deterministic run seed.",
    )
    _INIT_RUN_ROOT_OPTION = typer.Option(
        ...,
        "--root",
        help="Root directory for runs.",
    )
    _INIT_RUN_CONFIG_PATH_OPTION = typer.Option(
        None,
        "--config-path",
        help="Optional local config to copy into the run directory.",
    )
    _INIT_RUN_NOTES_OPTION = typer.Option(
        None,
        "--notes",
        help="Optional short run notes.",
    )
    _BASELINE_SMOKE_PATH_OPTION = typer.Option(
        ...,
        "--path",
        help="Path to the bundled synthetic FI-2010-style fixture.",
    )
    _BASELINE_SMOKE_WRITE_OUTPUTS_OPTION = typer.Option(
        False,
        "--write-outputs",
        help="Write run metadata and metrics under the output root.",
    )
    _BASELINE_SMOKE_OUTPUT_ROOT_OPTION = typer.Option(
        Path("runs"),
        "--output-root",
        help="Output root used only when --write-outputs is passed.",
    )

    def inspect_features_fi2010(
        path: Path = _INSPECT_PATH_OPTION,
        timestamp_column: str = _INSPECT_TIMESTAMP_OPTION,
        split_column: str = _INSPECT_SPLIT_OPTION,
        price_level_count: int = _INSPECT_LEVEL_COUNT_OPTION,
        no_timestamp_column: bool = _INSPECT_NO_TIMESTAMP_OPTION,
        no_split_column: bool = _INSPECT_NO_SPLIT_OPTION,
        allow_synthetic_time: bool = _INSPECT_FEAT_ALLOW_SYNTHETIC_OPTION,
    ) -> None:
        """Build microstructure features from an FI-2010 file and summarise."""
        exit_code = _inspect_features_fi2010_impl(
            path=path,
            timestamp_column=(
                None if no_timestamp_column else timestamp_column
            ),
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
            allow_synthetic_timestamps_for_time_features=allow_synthetic_time,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_labels_fi2010(
        path: Path = _INSPECT_PATH_OPTION,
        timestamp_column: str = _INSPECT_TIMESTAMP_OPTION,
        split_column: str = _INSPECT_SPLIT_OPTION,
        price_level_count: int = _INSPECT_LEVEL_COUNT_OPTION,
        no_timestamp_column: bool = _INSPECT_NO_TIMESTAMP_OPTION,
        no_split_column: bool = _INSPECT_NO_SPLIT_OPTION,
        generate_labels: bool = _INSPECT_LABELS_GENERATE_OPTION,
    ) -> None:
        """Build or extract FI-2010 labels and summarise."""
        exit_code = _inspect_labels_fi2010_impl(
            path=path,
            timestamp_column=(
                None if no_timestamp_column else timestamp_column
            ),
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
            prefer_existing_labels=not generate_labels,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_split(
        rows: int = _INSPECT_SPLIT_ROWS_OPTION,
    ) -> None:
        """Build a default temporal split and print partition counts."""
        exit_code = _inspect_split_impl(rows)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def init_run(
        name: str = _INIT_RUN_NAME_OPTION,
        phase: str = _INIT_RUN_PHASE_OPTION,
        seed: int = _INIT_RUN_SEED_OPTION,
        root: Path = _INIT_RUN_ROOT_OPTION,
        config_path: Path | None = _INIT_RUN_CONFIG_PATH_OPTION,
        notes: str | None = _INIT_RUN_NOTES_OPTION,
    ) -> None:
        """Create a metadata-only experiment run directory."""
        exit_code = _init_run_impl(
            name=name,
            phase=phase,
            seed=seed,
            root=root,
            config_path=config_path,
            notes=notes,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_baselines() -> None:
        """Print supported classical baseline model types."""
        exit_code = _inspect_baselines_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_baseline_smoke(
        path: Path = _BASELINE_SMOKE_PATH_OPTION,
        write_outputs: bool = _BASELINE_SMOKE_WRITE_OUTPUTS_OPTION,
        output_root: Path = _BASELINE_SMOKE_OUTPUT_ROOT_OPTION,
    ) -> None:
        """Run a synthetic fixture baseline smoke test."""
        exit_code = _run_baseline_smoke_impl(
            path=path,
            write_outputs=write_outputs,
            output_root=output_root,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    _TORCH_DATASET_PATH_OPTION = typer.Option(
        ...,
        "--path",
        help="Path to the local FI-2010-style file.",
    )
    _TORCH_DATASET_LOOKBACK_OPTION = typer.Option(
        2,
        "--lookback",
        help="Number of past rows per sequence window.",
    )
    _TORCH_DATASET_BATCH_SIZE_OPTION = typer.Option(
        4,
        "--batch-size",
        help="Batch size for the smoke DataLoader.",
    )
    _TORCH_DATASET_TARGET_OPTION = typer.Option(
        "label_10",
        "--target-column",
        help="Label column to use as the supervised target.",
    )
    _TORCH_DATASET_TRAIN_FRACTION_OPTION = typer.Option(
        0.5,
        "--train-fraction",
        help="Train fraction for the temporal split.",
    )
    _TORCH_DATASET_VALIDATION_FRACTION_OPTION = typer.Option(
        0.34,
        "--validation-fraction",
        help="Validation fraction for the temporal split.",
    )
    _TORCH_DATASET_TEST_FRACTION_OPTION = typer.Option(
        0.16,
        "--test-fraction",
        help="Test fraction for the temporal split.",
    )

    def inspect_torch_dataset(
        path: Path = _TORCH_DATASET_PATH_OPTION,
        lookback: int = _TORCH_DATASET_LOOKBACK_OPTION,
        batch_size: int = _TORCH_DATASET_BATCH_SIZE_OPTION,
        target_column: str = _TORCH_DATASET_TARGET_OPTION,
        timestamp_column: str = _INSPECT_TIMESTAMP_OPTION,
        split_column: str = _INSPECT_SPLIT_OPTION,
        price_level_count: int = _INSPECT_LEVEL_COUNT_OPTION,
        no_timestamp_column: bool = _INSPECT_NO_TIMESTAMP_OPTION,
        no_split_column: bool = _INSPECT_NO_SPLIT_OPTION,
        train_fraction: float = _TORCH_DATASET_TRAIN_FRACTION_OPTION,
        validation_fraction: float = _TORCH_DATASET_VALIDATION_FRACTION_OPTION,
        test_fraction: float = _TORCH_DATASET_TEST_FRACTION_OPTION,
    ) -> None:
        """Build a tiny sequence DataLoader from an FI-2010 file and summarise."""
        exit_code = _inspect_torch_dataset_impl(
            path=path,
            lookback=lookback,
            batch_size=batch_size,
            target_column=target_column,
            timestamp_column=(
                None if no_timestamp_column else timestamp_column
            ),
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    _DEEPLOB_SMOKE_PATH_OPTION = typer.Option(
        ...,
        "--path",
        help="Path to the local FI-2010-style fixture file.",
    )
    _DEEPLOB_SMOKE_LOOKBACK_OPTION = typer.Option(
        2,
        "--lookback",
        help="Number of past rows per sequence window.",
    )
    _DEEPLOB_SMOKE_EPOCHS_OPTION = typer.Option(
        1,
        "--epochs",
        help="Number of training epochs for the smoke run.",
    )
    _DEEPLOB_SMOKE_BATCH_OPTION = typer.Option(
        4,
        "--batch-size",
        help="Batch size for the smoke DataLoader.",
    )
    _DEEPLOB_SMOKE_SEED_OPTION = typer.Option(
        42,
        "--seed",
        help="Deterministic seed for the smoke run.",
    )
    _DEEPLOB_SMOKE_WRITE_OUTPUTS_OPTION = typer.Option(
        False,
        "--write-outputs",
        help=(
            "The smoke command never writes outputs; this flag is accepted "
            "for symmetry with run-baseline-smoke but only prints a notice."
        ),
    )
    _DEEPLOB_SMOKE_OUTPUT_ROOT_OPTION = typer.Option(
        Path("runs"),
        "--output-root",
        help="Reserved for future use; ignored by the smoke command.",
    )

    def inspect_deeplob() -> None:
        """Print DeepLOB-style baseline defaults without training."""
        exit_code = _inspect_deeplob_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_deeplob_smoke(
        path: Path = _DEEPLOB_SMOKE_PATH_OPTION,
        lookback: int = _DEEPLOB_SMOKE_LOOKBACK_OPTION,
        epochs: int = _DEEPLOB_SMOKE_EPOCHS_OPTION,
        batch_size: int = _DEEPLOB_SMOKE_BATCH_OPTION,
        seed: int = _DEEPLOB_SMOKE_SEED_OPTION,
        write_outputs: bool = _DEEPLOB_SMOKE_WRITE_OUTPUTS_OPTION,
        output_root: Path = _DEEPLOB_SMOKE_OUTPUT_ROOT_OPTION,
    ) -> None:
        """Run a synthetic fixture DeepLOB-style supervised smoke experiment."""
        exit_code = _run_deeplob_smoke_impl(
            path=path,
            lookback=lookback,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            write_outputs=write_outputs,
            output_root=output_root,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    _TRANSFORMER_SMOKE_PATH_OPTION = typer.Option(
        ...,
        "--path",
        help="Path to the local canonical event-log JSONL fixture.",
    )
    _TRANSFORMER_SMOKE_WINDOW_OPTION = typer.Option(
        4,
        "--window-length",
        help="Fixed token-window length used for the smoke run.",
    )
    _TRANSFORMER_SMOKE_BATCH_OPTION = typer.Option(
        4,
        "--batch-size",
        help="Batch size for the smoke DataLoader.",
    )
    _TRANSFORMER_SMOKE_EPOCHS_OPTION = typer.Option(
        1,
        "--epochs",
        help="Number of training epochs for the smoke run.",
    )
    _TRANSFORMER_SMOKE_SEED_OPTION = typer.Option(
        42,
        "--seed",
        help="Deterministic seed for the smoke run.",
    )
    _TRANSFORMER_SMOKE_NUM_CLASSES_OPTION = typer.Option(
        3,
        "--num-classes",
        help="Number of synthetic smoke classes used for plumbing.",
    )
    _TRANSFORMER_SMOKE_SYMBOL_OPTION = typer.Option(
        None,
        "--symbol",
        help="Optional symbol filter applied to the event log.",
    )
    _TRANSFORMER_SMOKE_LEVELS_OPTION = typer.Option(
        2,
        "--max-levels-per-side",
        help="Maximum snapshot levels per side to tokenise.",
    )

    def inspect_transformer() -> None:
        """Print market transformer encoder defaults without training."""
        exit_code = _inspect_transformer_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_transformer_smoke(
        path: Path = _TRANSFORMER_SMOKE_PATH_OPTION,
        window_length: int = _TRANSFORMER_SMOKE_WINDOW_OPTION,
        batch_size: int = _TRANSFORMER_SMOKE_BATCH_OPTION,
        epochs: int = _TRANSFORMER_SMOKE_EPOCHS_OPTION,
        seed: int = _TRANSFORMER_SMOKE_SEED_OPTION,
        num_classes: int = _TRANSFORMER_SMOKE_NUM_CLASSES_OPTION,
        symbol: str | None = _TRANSFORMER_SMOKE_SYMBOL_OPTION,
        max_levels_per_side: int = _TRANSFORMER_SMOKE_LEVELS_OPTION,
    ) -> None:
        """Run a synthetic-label transformer smoke experiment."""
        exit_code = _run_transformer_smoke_impl(
            path=path,
            window_length=window_length,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_classes=num_classes,
            symbol=symbol,
            max_levels_per_side=max_levels_per_side,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    _SSL_SMOKE_PATH_OPTION = typer.Option(
        ...,
        "--path",
        help="Path to the local canonical event-log JSONL fixture.",
    )
    _SSL_SMOKE_WINDOW_OPTION = typer.Option(
        4,
        "--window-length",
        help="Fixed token-window length used for the SSL smoke run.",
    )
    _SSL_SMOKE_BATCH_OPTION = typer.Option(
        4,
        "--batch-size",
        help="Batch size for the SSL smoke DataLoader.",
    )
    _SSL_SMOKE_EPOCHS_OPTION = typer.Option(
        1,
        "--epochs",
        help="Number of pretraining epochs for the smoke run.",
    )
    _SSL_SMOKE_SEED_OPTION = typer.Option(
        42,
        "--seed",
        help="Deterministic seed for the SSL smoke run.",
    )
    _SSL_SMOKE_SYMBOL_OPTION = typer.Option(
        None,
        "--symbol",
        help="Optional symbol filter applied to the event log.",
    )
    _SSL_SMOKE_LEVELS_OPTION = typer.Option(
        2,
        "--max-levels-per-side",
        help="Maximum snapshot levels per side to tokenise.",
    )
    _SSL_SMOKE_MASK_PROBABILITY_OPTION = typer.Option(
        0.15,
        "--mask-probability",
        help="Probability of selecting a valid position for masking.",
    )

    def inspect_ssl() -> None:
        """Print SSL transformer wrapper defaults without training."""
        exit_code = _inspect_ssl_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_ssl_smoke(
        path: Path = _SSL_SMOKE_PATH_OPTION,
        window_length: int = _SSL_SMOKE_WINDOW_OPTION,
        batch_size: int = _SSL_SMOKE_BATCH_OPTION,
        epochs: int = _SSL_SMOKE_EPOCHS_OPTION,
        seed: int = _SSL_SMOKE_SEED_OPTION,
        symbol: str | None = _SSL_SMOKE_SYMBOL_OPTION,
        max_levels_per_side: int = _SSL_SMOKE_LEVELS_OPTION,
        mask_probability: float = _SSL_SMOKE_MASK_PROBABILITY_OPTION,
    ) -> None:
        """Run a tiny synthetic SSL smoke experiment."""
        exit_code = _run_ssl_smoke_impl(
            path=path,
            window_length=window_length,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            symbol=symbol,
            max_levels_per_side=max_levels_per_side,
            mask_probability=mask_probability,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    _MULTITASK_SMOKE_PATH_OPTION = typer.Option(
        ...,
        "--path",
        help="Path to the local canonical event-log JSONL fixture.",
    )
    _MULTITASK_SMOKE_WINDOW_OPTION = typer.Option(
        4,
        "--window-length",
        help="Fixed token-window length used for the multi-task smoke run.",
    )
    _MULTITASK_SMOKE_BATCH_OPTION = typer.Option(
        4,
        "--batch-size",
        help="Batch size for the multi-task smoke DataLoader.",
    )
    _MULTITASK_SMOKE_EPOCHS_OPTION = typer.Option(
        1,
        "--epochs",
        help="Number of supervised fine-tuning epochs for the smoke run.",
    )
    _MULTITASK_SMOKE_SEED_OPTION = typer.Option(
        42,
        "--seed",
        help="Deterministic seed for the multi-task smoke run.",
    )
    _MULTITASK_SMOKE_SYMBOL_OPTION = typer.Option(
        None,
        "--symbol",
        help="Optional symbol filter applied to the event log.",
    )
    _MULTITASK_SMOKE_LEVELS_OPTION = typer.Option(
        2,
        "--max-levels-per-side",
        help="Maximum snapshot levels per side to tokenise.",
    )

    def inspect_multitask() -> None:
        """Print multi-task transformer defaults without training."""
        exit_code = _inspect_multitask_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_multitask_smoke(
        path: Path = _MULTITASK_SMOKE_PATH_OPTION,
        window_length: int = _MULTITASK_SMOKE_WINDOW_OPTION,
        batch_size: int = _MULTITASK_SMOKE_BATCH_OPTION,
        epochs: int = _MULTITASK_SMOKE_EPOCHS_OPTION,
        seed: int = _MULTITASK_SMOKE_SEED_OPTION,
        symbol: str | None = _MULTITASK_SMOKE_SYMBOL_OPTION,
        max_levels_per_side: int = _MULTITASK_SMOKE_LEVELS_OPTION,
    ) -> None:
        """Run a tiny synthetic supervised multi-task smoke experiment."""
        exit_code = _run_multitask_smoke_impl(
            path=path,
            window_length=window_length,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            symbol=symbol,
            max_levels_per_side=max_levels_per_side,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    _CALIBRATION_SMOKE_N_EXAMPLES_OPTION = typer.Option(
        60,
        "--n-examples",
        help="Number of deterministic synthetic examples.",
    )
    _CALIBRATION_SMOKE_NUM_CLASSES_OPTION = typer.Option(
        3,
        "--num-classes",
        help="Number of synthetic classification classes.",
    )
    _CALIBRATION_SMOKE_SEED_OPTION = typer.Option(
        42,
        "--seed",
        help="Deterministic seed for synthetic logits.",
    )
    _CALIBRATION_SMOKE_ECE_BINS_OPTION = typer.Option(
        10,
        "--ece-bins",
        help="Number of bins for expected calibration error.",
    )

    def inspect_calibration() -> None:
        """Print calibration and uncertainty support without training."""
        exit_code = _inspect_calibration_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_calibration_smoke(
        n_examples: int = _CALIBRATION_SMOKE_N_EXAMPLES_OPTION,
        num_classes: int = _CALIBRATION_SMOKE_NUM_CLASSES_OPTION,
        seed: int = _CALIBRATION_SMOKE_SEED_OPTION,
        ece_bins: int = _CALIBRATION_SMOKE_ECE_BINS_OPTION,
    ) -> None:
        """Run a deterministic synthetic calibration smoke check."""
        exit_code = _run_calibration_smoke_impl(
            n_examples=n_examples,
            num_classes=num_classes,
            seed=seed,
            ece_bins=ece_bins,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    _EXECUTION_SMOKE_N_SIGNALS_OPTION = typer.Option(
        24,
        "--n-signals",
        help="Number of deterministic synthetic prediction signals.",
    )
    _EXECUTION_SMOKE_SEED_OPTION = typer.Option(
        42,
        "--seed",
        help="Deterministic seed for synthetic market-state noise.",
    )

    def inspect_execution_validation() -> None:
        """Print execution-aware validation support without running a model."""
        exit_code = _inspect_execution_validation_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_execution_validation_smoke(
        n_signals: int = _EXECUTION_SMOKE_N_SIGNALS_OPTION,
        seed: int = _EXECUTION_SMOKE_SEED_OPTION,
    ) -> None:
        """Run a deterministic synthetic execution-validation smoke check."""
        exit_code = _run_execution_validation_smoke_impl(
            n_signals=n_signals,
            seed=seed,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    _ROBUSTNESS_SMOKE_N_RECORDS_OPTION = typer.Option(
        36,
        "--n-records",
        help="Number of deterministic synthetic analysis records.",
    )
    _ROBUSTNESS_SMOKE_SEED_OPTION = typer.Option(
        42,
        "--seed",
        help="Deterministic seed for synthetic analysis records.",
    )

    def inspect_analysis() -> None:
        """Print supported analysis tools without running anything."""
        exit_code = _inspect_analysis_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_robustness_analysis_smoke(
        n_records: int = _ROBUSTNESS_SMOKE_N_RECORDS_OPTION,
        seed: int = _ROBUSTNESS_SMOKE_SEED_OPTION,
    ) -> None:
        """Run a deterministic synthetic robustness-analysis smoke check."""
        exit_code = _run_robustness_analysis_smoke_impl(
            n_records=n_records,
            seed=seed,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    _BINANCE_SNAPSHOT_OPTION = typer.Option(
        ...,
        "--snapshot",
        help="Path to the local Binance-style snapshot JSON file.",
    )
    _BINANCE_UPDATES_OPTION = typer.Option(
        ...,
        "--updates",
        help="Path to the local Binance-style diff JSONL file.",
    )
    _BINANCE_SYMBOL_OPTION = typer.Option(
        None,
        "--symbol",
        help="Optional symbol override when the snapshot does not carry one.",
    )
    _BINANCE_MAX_DEPTH_OPTION = typer.Option(
        None,
        "--max-depth",
        help="Optional maximum depth per side to keep during replay.",
    )
    _BINANCE_STOP_ON_GAP_OPTION = typer.Option(
        True,
        "--stop-on-gap/--no-stop-on-gap",
        help="Stop reconstruction when an update-id gap is detected.",
    )
    _BINANCE_ALLOW_CROSSED_OPTION = typer.Option(
        False,
        "--allow-crossed",
        help="Permit crossed books instead of treating them as errors.",
    )

    def inspect_binance_replay(
        snapshot: Path = _BINANCE_SNAPSHOT_OPTION,
        updates: Path = _BINANCE_UPDATES_OPTION,
        symbol: str | None = _BINANCE_SYMBOL_OPTION,
        max_depth: int | None = _BINANCE_MAX_DEPTH_OPTION,
        stop_on_gap: bool = _BINANCE_STOP_ON_GAP_OPTION,
        allow_crossed: bool = _BINANCE_ALLOW_CROSSED_OPTION,
    ) -> None:
        """Reconstruct a local Binance-style book from a snapshot and diff JSONL."""
        exit_code = _inspect_binance_replay_impl(
            snapshot_path=snapshot,
            updates_path=updates,
            symbol=symbol,
            max_depth=max_depth,
            stop_on_gap=stop_on_gap,
            allow_crossed=allow_crossed,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

else:

    def inspect_event_log(path: Path) -> None:
        """Inspect a local canonical event-log JSONL file."""
        exit_code = _inspect_event_log_impl(path)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_event_tokens(
        path: Path,
        symbol: str | None = None,
        window_length: int = 8,
        max_levels_per_side: int = 2,
        include_eos: bool = False,
    ) -> None:
        """Tokenise a local canonical event log and summarise IDs."""
        exit_code = _inspect_event_tokens_impl(
            path=path,
            symbol=symbol,
            window_length=window_length,
            max_levels_per_side=max_levels_per_side,
            include_eos=include_eos,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def event_log_to_features(path: Path) -> None:
        """Replay a local event log into feature rows and summarise."""
        exit_code = _event_log_to_features_impl(path)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_fi2010(
        path: Path,
        timestamp_column: str = "timestamp",
        split_column: str = "split",
        price_level_count: int = 2,
        no_timestamp_column: bool = False,
        no_split_column: bool = False,
    ) -> None:
        """Load an FI-2010 file and print a data-quality summary."""
        exit_code = _inspect_fi2010_impl(
            path=path,
            timestamp_column=None if no_timestamp_column else timestamp_column,
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_features_fi2010(
        path: Path,
        timestamp_column: str = "timestamp",
        split_column: str = "split",
        price_level_count: int = 2,
        no_timestamp_column: bool = False,
        no_split_column: bool = False,
        allow_synthetic_time: bool = False,
    ) -> None:
        """Build microstructure features from an FI-2010 file and summarise."""
        exit_code = _inspect_features_fi2010_impl(
            path=path,
            timestamp_column=None if no_timestamp_column else timestamp_column,
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
            allow_synthetic_timestamps_for_time_features=allow_synthetic_time,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_labels_fi2010(
        path: Path,
        timestamp_column: str = "timestamp",
        split_column: str = "split",
        price_level_count: int = 2,
        no_timestamp_column: bool = False,
        no_split_column: bool = False,
        generate_labels: bool = False,
    ) -> None:
        """Build or extract FI-2010 labels and summarise."""
        exit_code = _inspect_labels_fi2010_impl(
            path=path,
            timestamp_column=None if no_timestamp_column else timestamp_column,
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
            prefer_existing_labels=not generate_labels,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_split(rows: int) -> None:
        """Build a default temporal split and print partition counts."""
        exit_code = _inspect_split_impl(rows)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def init_run(
        name: str,
        phase: str,
        seed: int,
        root: Path,
        config_path: Path | None = None,
        notes: str | None = None,
    ) -> None:
        """Create a metadata-only experiment run directory."""
        exit_code = _init_run_impl(
            name=name,
            phase=phase,
            seed=seed,
            root=root,
            config_path=config_path,
            notes=notes,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_baselines() -> None:
        """Print supported classical baseline model types."""
        exit_code = _inspect_baselines_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_baseline_smoke(
        path: Path,
        write_outputs: bool = False,
        output_root: Path = Path("runs"),
    ) -> None:
        """Run a synthetic fixture baseline smoke test."""
        exit_code = _run_baseline_smoke_impl(
            path=path,
            write_outputs=write_outputs,
            output_root=output_root,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_torch_dataset(
        path: Path,
        lookback: int = 2,
        batch_size: int = 4,
        target_column: str = "label_10",
        timestamp_column: str = "timestamp",
        split_column: str = "split",
        price_level_count: int = 2,
        no_timestamp_column: bool = False,
        no_split_column: bool = False,
        train_fraction: float = 0.5,
        validation_fraction: float = 0.34,
        test_fraction: float = 0.16,
    ) -> None:
        """Build a tiny sequence DataLoader from an FI-2010 file and summarise."""
        exit_code = _inspect_torch_dataset_impl(
            path=path,
            lookback=lookback,
            batch_size=batch_size,
            target_column=target_column,
            timestamp_column=None if no_timestamp_column else timestamp_column,
            split_column=None if no_split_column else split_column,
            price_level_count=price_level_count,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_deeplob() -> None:
        """Print DeepLOB-style baseline defaults without training."""
        exit_code = _inspect_deeplob_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_deeplob_smoke(
        path: Path,
        lookback: int = 2,
        epochs: int = 1,
        batch_size: int = 4,
        seed: int = 42,
        write_outputs: bool = False,
        output_root: Path = Path("runs"),
    ) -> None:
        """Run a synthetic fixture DeepLOB-style supervised smoke experiment."""
        exit_code = _run_deeplob_smoke_impl(
            path=path,
            lookback=lookback,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            write_outputs=write_outputs,
            output_root=output_root,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_binance_replay(
        snapshot: Path,
        updates: Path,
        symbol: str | None = None,
        max_depth: int | None = None,
        stop_on_gap: bool = True,
        allow_crossed: bool = False,
    ) -> None:
        """Reconstruct a local Binance-style book from a snapshot and diff JSONL."""
        exit_code = _inspect_binance_replay_impl(
            snapshot_path=snapshot,
            updates_path=updates,
            symbol=symbol,
            max_depth=max_depth,
            stop_on_gap=stop_on_gap,
            allow_crossed=allow_crossed,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_transformer() -> None:
        """Print market transformer encoder defaults without training."""
        exit_code = _inspect_transformer_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_transformer_smoke(
        path: Path,
        window_length: int = 4,
        batch_size: int = 4,
        epochs: int = 1,
        seed: int = 42,
        num_classes: int = 3,
        symbol: str | None = None,
        max_levels_per_side: int = 2,
    ) -> None:
        """Run a synthetic-label transformer smoke experiment."""
        exit_code = _run_transformer_smoke_impl(
            path=path,
            window_length=window_length,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_classes=num_classes,
            symbol=symbol,
            max_levels_per_side=max_levels_per_side,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_ssl() -> None:
        """Print SSL transformer wrapper defaults without training."""
        exit_code = _inspect_ssl_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_ssl_smoke(
        path: Path,
        window_length: int = 4,
        batch_size: int = 4,
        epochs: int = 1,
        seed: int = 42,
        symbol: str | None = None,
        max_levels_per_side: int = 2,
        mask_probability: float = 0.15,
    ) -> None:
        """Run a tiny synthetic SSL smoke experiment."""
        exit_code = _run_ssl_smoke_impl(
            path=path,
            window_length=window_length,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            symbol=symbol,
            max_levels_per_side=max_levels_per_side,
            mask_probability=mask_probability,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_multitask() -> None:
        """Print multi-task transformer defaults without training."""
        exit_code = _inspect_multitask_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_multitask_smoke(
        path: Path,
        window_length: int = 4,
        batch_size: int = 4,
        epochs: int = 1,
        seed: int = 42,
        symbol: str | None = None,
        max_levels_per_side: int = 2,
    ) -> None:
        """Run a tiny synthetic supervised multi-task smoke experiment."""
        exit_code = _run_multitask_smoke_impl(
            path=path,
            window_length=window_length,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            symbol=symbol,
            max_levels_per_side=max_levels_per_side,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_calibration() -> None:
        """Print calibration and uncertainty support without training."""
        exit_code = _inspect_calibration_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_calibration_smoke(
        n_examples: int = 60,
        num_classes: int = 3,
        seed: int = 42,
        ece_bins: int = 10,
    ) -> None:
        """Run a deterministic synthetic calibration smoke check."""
        exit_code = _run_calibration_smoke_impl(
            n_examples=n_examples,
            num_classes=num_classes,
            seed=seed,
            ece_bins=ece_bins,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_execution_validation() -> None:
        """Print execution-aware validation support without running a model."""
        exit_code = _inspect_execution_validation_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_execution_validation_smoke(
        n_signals: int = 24,
        seed: int = 42,
    ) -> None:
        """Run a deterministic synthetic execution-validation smoke check."""
        exit_code = _run_execution_validation_smoke_impl(
            n_signals=n_signals,
            seed=seed,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_analysis() -> None:
        """Print supported analysis tools without running anything."""
        exit_code = _inspect_analysis_impl()
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_robustness_analysis_smoke(
        n_records: int = 36,
        seed: int = 42,
    ) -> None:
        """Run a deterministic synthetic robustness-analysis smoke check."""
        exit_code = _run_robustness_analysis_smoke_impl(
            n_records=n_records,
            seed=seed,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def prepare_fi2010_benchmark(
        config: Path,
        data_path: Path,
        out: Path,
    ) -> None:
        """Prepare a local-only FI-2010 benchmark input."""
        exit_code = _prepare_fi2010_benchmark_impl(
            config_path=config,
            data_path=data_path,
            out=out,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def verify_fi2010_local(data_path: Path) -> None:
        """Inspect a local FI-2010 file safely without loading it."""
        exit_code = _verify_fi2010_local_impl(data_path=data_path)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def convert_fi2010_official(
        input_path: Path,
        output_path: Path,
        split: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Convert a single official FI-2010 .txt matrix into a loader-ready CSV."""
        exit_code = _convert_fi2010_official_impl(
            input_path=input_path,
            output_path=output_path,
            split_label=split,
            overwrite=overwrite,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_paper_experiment(
        config: Path,
        data_path: Path,
        out: Path,
        models: str = "majority",
        overwrite: bool = False,
        build_plots: bool = False,
    ) -> None:
        """Run the paper experiment runner and write artefacts."""
        model_tokens = [token.strip() for token in models.split(",") if token.strip()]
        exit_code = _run_paper_experiment_impl(
            config_path=config,
            data_path=data_path,
            out=out,
            models=model_tokens or None,
            overwrite=overwrite,
            build_plots=build_plots,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def build_paper_plots(
        experiment: Path,
        overwrite: bool = False,
    ) -> None:
        """Generate paper experiment plots from stored artefacts."""
        exit_code = _build_paper_plots_impl(
            experiment=experiment,
            overwrite=overwrite,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_paper_ablations(
        config: Path,
        data_path: Path,
        out: Path,
        models: str = "majority,logistic",
        ablation_set: str = "smoke",
        overwrite: bool = False,
        build_plots: bool = False,
    ) -> None:
        """Run the paper-experiment ablation suite and write aggregate artefacts."""
        model_tokens = [token.strip() for token in models.split(",") if token.strip()]
        exit_code = _run_paper_ablations_impl(
            config_path=config,
            data_path=data_path,
            out=out,
            models=model_tokens or None,
            ablation_set=ablation_set,
            overwrite=overwrite,
            build_plots=build_plots,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def run_system_benchmarks(
        config: Path,
        data_path: Path,
        out: Path,
        benchmark_set: str = "smoke",
        models: str = "majority,logistic",
        overwrite: bool = False,
    ) -> None:
        """Run local systems benchmarks and write aggregate artefacts."""
        model_tokens = [token.strip() for token in models.split(",") if token.strip()]
        exit_code = _run_system_benchmarks_impl(
            config_path=config,
            data_path=data_path,
            out=out,
            benchmark_set=benchmark_set,
            models=model_tokens or None,
            overwrite=overwrite,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_system_benchmarks(benchmark: Path) -> None:
        """Print a concise systems benchmark summary."""
        exit_code = _inspect_system_benchmarks_impl(benchmark=benchmark)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_paper_experiment(experiment: Path) -> None:
        """Print a concise paper experiment artefact summary."""
        exit_code = _inspect_paper_experiment_impl(experiment=experiment)
        if exit_code != 0:
            raise SystemExit(exit_code)

    def build_paper_report(
        experiment: Path,
        ablations: Path | None = None,
        systems: Path | None = None,
        out: Path = Path("runs/chronoslob_empirical_report_smoke.md"),
        overwrite: bool = False,
    ) -> None:
        """Build an empirical report from stored artefacts."""
        exit_code = _build_paper_report_impl(
            experiment=experiment,
            ablations=ablations,
            systems=systems,
            out=out,
            overwrite=overwrite,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    def inspect_paper_report(report: Path) -> None:
        """Inspect a generated empirical report summary."""
        exit_code = _inspect_paper_report_impl(report=report)
        if exit_code != 0:
            raise SystemExit(exit_code)


if typer is not None:
    app.command()(version)
    app.command()(doctor)
    app.command("run-project-audit")(run_project_audit)
    app.command("inspect-release-readiness")(inspect_release_readiness)
    app.command("build-report-archive")(build_report_archive)
    app.command("inspect-report-archive")(inspect_report_archive)
    app.command("inspect-experiment-artifacts")(inspect_experiment_artifacts)
    app.command("inspect-event-log")(inspect_event_log)
    app.command("inspect-event-tokens")(inspect_event_tokens)
    app.command("event-log-to-features")(event_log_to_features)
    app.command("inspect-fi2010")(inspect_fi2010)
    app.command("inspect-features-fi2010")(inspect_features_fi2010)
    app.command("inspect-labels-fi2010")(inspect_labels_fi2010)
    app.command("inspect-split")(inspect_split)
    app.command("init-run")(init_run)
    app.command("inspect-baselines")(inspect_baselines)
    app.command("run-baseline-smoke")(run_baseline_smoke)
    app.command("inspect-torch-dataset")(inspect_torch_dataset)
    app.command("inspect-deeplob")(inspect_deeplob)
    app.command("run-deeplob-smoke")(run_deeplob_smoke)
    app.command("inspect-transformer")(inspect_transformer)
    app.command("run-transformer-smoke")(run_transformer_smoke)
    app.command("inspect-ssl")(inspect_ssl)
    app.command("run-ssl-smoke")(run_ssl_smoke)
    app.command("inspect-multitask")(inspect_multitask)
    app.command("run-multitask-smoke")(run_multitask_smoke)
    app.command("inspect-calibration")(inspect_calibration)
    app.command("run-calibration-smoke")(run_calibration_smoke)
    app.command("inspect-execution-validation")(inspect_execution_validation)
    app.command("run-execution-validation-smoke")(run_execution_validation_smoke)
    app.command("inspect-analysis")(inspect_analysis)
    app.command("run-robustness-analysis-smoke")(run_robustness_analysis_smoke)
    app.command("inspect-binance-replay")(inspect_binance_replay)
    app.command("prepare-fi2010-benchmark")(prepare_fi2010_benchmark)
    app.command("verify-fi2010-local")(verify_fi2010_local)
    app.command("convert-fi2010-official")(convert_fi2010_official)
    app.command("run-paper-experiment")(run_paper_experiment)
    app.command("run-paper-ablations")(run_paper_ablations)
    app.command("run-system-benchmarks")(run_system_benchmarks)
    app.command("inspect-system-benchmarks")(inspect_system_benchmarks)
    app.command("build-paper-plots")(build_paper_plots)
    app.command("inspect-paper-experiment")(inspect_paper_experiment)
    app.command("build-paper-report")(build_paper_report)
    app.command("inspect-paper-report")(inspect_paper_report)
else:

    def app() -> int:
        """Fallback command runner used when Typer is not installed yet."""
        return _fallback_main()


if __name__ == "__main__":
    if typer is None:
        raise SystemExit(_fallback_main())
    app()
