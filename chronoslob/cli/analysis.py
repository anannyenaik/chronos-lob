"""Analysis and reporting command implementations."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _analyse_fi2010_feature_ablations_impl(
    *,
    feature_ablations: Path,
    extra_feature_ablations: str | None,
    out: Path,
    figures: bool,
    overwrite: bool,
    allow_smoke_test: bool,
) -> int:
    """Build feature-ablation stability analysis from lightweight artefacts."""
    from chronoslob.analysis.fi2010_feature_ablation_analysis import (
        analyse_fi2010_feature_ablations,
    )

    try:
        summary = analyse_fi2010_feature_ablations(
            ablation_dir=feature_ablations,
            extra_ablation_dirs=extra_feature_ablations,
            out_dir=out,
            figures=figures,
            overwrite=overwrite,
            allow_smoke_test=allow_smoke_test,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"FI-2010 feature-ablation analysis failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 feature-ablation stability analysis")
    print(f"  feature ablations:   {summary.ablation_dir}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  evidence status:     {summary.evidence_status}")
    print(f"  completed runs:      {summary.completed_run_count}")
    print(f"  failed runs:         {summary.failed_run_count}")
    print(f"  horizons:            {summary.horizons}")
    print(f"  models:              {', '.join(summary.models)}")
    print(
        "  raw predictions:     "
        f"{'available' if summary.raw_predictions_available else 'not available'}"
    )
    print(f"  completed figures:   {len(summary.figures_completed)}")
    print(f"  skipped figures:     {len(summary.figures_skipped)}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  files written:")
    for key, path in summary.files_written.items():
        print(f"    {key}: {path}")
    print("  network calls:       none performed")
    return 0


def _analyse_fi2010_uncertainty_impl(
    *,
    classical_dir: Path | None,
    neural_dir: Path | None,
    out: Path,
    baseline_model: str,
    ci_level: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    overwrite: bool,
) -> int:
    """Compute uncertainty artefacts from stored multi-fold tables."""
    from chronoslob.experiments.statistics import (
        DEFAULT_UNCERTAINTY_METRICS,
        analyse_fi2010_uncertainty,
    )

    try:
        summary = analyse_fi2010_uncertainty(
            classical_dir=classical_dir,
            neural_dir=neural_dir,
            out_dir=out,
            baseline_model=baseline_model,
            metrics=DEFAULT_UNCERTAINTY_METRICS,
            ci_level=ci_level,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"FI-2010 uncertainty analysis failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 uncertainty analysis")
    print(f"  classical input:     {summary.classical_input}")
    print(f"  neural input:        {summary.neural_input}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  baseline model:      {summary.baseline_model}")
    print(f"  ci level:            {summary.ci_level}")
    print(f"  bootstrap iterations: {summary.bootstrap_iterations}")
    print(f"  bootstrap seed:      {summary.bootstrap_seed}")
    print(f"  metrics:             {', '.join(summary.metrics)}")
    print(
        "  classical models:    "
        + (", ".join(summary.classical_models) if summary.classical_models else "none")
    )
    print(
        "  neural models:       "
        + (", ".join(summary.neural_models) if summary.neural_models else "none")
    )
    print(
        "  classical folds:     "
        + (", ".join(summary.classical_folds) if summary.classical_folds else "none")
    )
    print(
        "  neural folds:        "
        + (", ".join(summary.neural_folds) if summary.neural_folds else "none")
    )
    print(
        "  neural seeds:        "
        + (
            ", ".join(str(value) for value in summary.neural_seeds)
            if summary.neural_seeds
            else "none"
        )
    )
    print(
        "  neural lookbacks:    "
        + (
            ", ".join(str(value) for value in summary.neural_lookbacks)
            if summary.neural_lookbacks
            else "none"
        )
    )
    print(
        f"  classical seed variance: {'yes' if summary.classical_seed_variance_available else 'no'}"
    )
    print(f"  neural seed variance:    {'yes' if summary.neural_seed_variance_available else 'no'}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  predictions:         not required")
    print("  checkpoints:         not required")
    print("  network calls:       none performed")
    return 0


def _analyse_fi2010_ssl_v2_results_impl(
    *,
    ssl_v2_dir: Path,
    out: Path,
) -> int:
    """Analyse stored FI-2010 SSL-v2 benchmark artefacts and write a report.

    Stored summary tables are always read. Per-run predictions, when
    present in the benchmark ``runs/`` tree, additionally enable the
    confidence-filtered diagnostics; raw prediction files are not
    required.
    """
    from chronoslob.analysis.ssl_v2_analysis import analyse_ssl_v2_results

    try:
        summary = analyse_ssl_v2_results(ssl_v2_dir, out_dir=out)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"FI-2010 SSL-v2 analysis failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 SSL-v2 analysis")
    print(f"  benchmark input:        {summary.ssl_v2_dir}")
    print(f"  output directory:       {summary.out_dir}")
    print(f"  result coverage:        {summary.evidence_level}")
    print(f"  matched SSL-v2 rows:    {summary.ssl_v2_matched_rows}")
    print(f"  confidence rows:        {summary.confidence_filtered_rows}")
    print(f"  active-fraction proxy:  {summary.execution_proxy_available}")
    print(f"  failures:               {summary.failure_count}")
    for claim_id, status in sorted(summary.claim_statuses.items()):
        print(f"    finding {claim_id}: {status}")
    print("  network calls:          none performed")
    return 0


def _analyse_fi2010_ssl_results_impl(
    *,
    full_grid_dir: Path | None,
    proper_training_dir: Path | None,
    out: Path,
    make_figures: bool,
    overwrite: bool,
) -> int:
    """Analyse stored FI-2010 SSL comparison artefacts and write a report.

    Only stored summary tables are read; unavailable raw prediction
    files and encoder checkpoints are never required.
    """
    from chronoslob.analysis.ssl_failure_analysis import analyse_fi2010_ssl_results

    try:
        summary = analyse_fi2010_ssl_results(
            full_grid_dir=full_grid_dir,
            proper_training_dir=proper_training_dir,
            out_dir=out,
            make_figures=make_figures,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"FI-2010 SSL analysis failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 SSL failure analysis")
    print(f"  full grid input:     {summary.full_grid_dir}")
    print(f"  validation-selected: {summary.proper_training_dir}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  full-grid matched rows:       {summary.full_grid_matched_rows}")
    print(f"  validation-selected matched rows: {summary.proper_training_matched_rows}")
    print("  interpretation:")
    for claim_id, status in summary.claim_statuses.items():
        print(f"    {claim_id}: {status}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    if summary.figures_generated:
        print("  figures:             " + ", ".join(summary.figures_generated))
    else:
        print("  figures:             none")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  raw predictions:     not required")
    print("  checkpoints:         not required")
    print("  network calls:       none performed")
    return 0


def _analyse_fi2010_execution_v3_impl(
    *,
    execution_v3_dir: Path,
    out: Path,
    make_figures: bool,
    overwrite: bool,
) -> int:
    """Build the execution-v3 proxy analysis from stored summary tables.

    Only stored execution-v3 summary tables are read; unavailable raw
    prediction arrays are never required.
    """
    from chronoslob.analysis.execution_v3_analysis import analyse_fi2010_execution_v3

    try:
        summary = analyse_fi2010_execution_v3(
            execution_v3_dir=execution_v3_dir,
            out_dir=out,
            make_figures=make_figures,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"FI-2010 execution-v3 analysis failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 execution-v3 proxy analysis")
    print(f"  execution-v3 input:  {summary.execution_v3_dir}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  payoff/cost mode:    {summary.payoff_mode}/{summary.cost_mode}")
    print(f"  run groups:          {summary.run_group_count}")
    print(f"  regime diagnostics:  {summary.regime_status}")
    print("  interpretation:")
    for claim_id, status in summary.claim_statuses.items():
        print(f"    {claim_id}: {status}")
    if summary.figures_generated:
        print("  figures:             " + ", ".join(summary.figures_generated))
    else:
        print("  figures:             none")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  raw predictions:     not required")
    print("  network calls:       none performed")
    return 0


def _build_execution_centrepiece_impl(
    *,
    execution_analysis: Path,
    out: Path,
    execution_v3: Path | None,
    neural_full_grid: Path | None,
    make_figures: bool,
    overwrite: bool,
) -> int:
    """Build the forecasting-versus-signal-quality execution centrepiece."""
    from chronoslob.analysis.execution_centrepiece import build_execution_centrepiece

    try:
        summary = build_execution_centrepiece(
            execution_analysis_dir=execution_analysis,
            out_dir=out,
            execution_v3_dir=execution_v3,
            neural_full_grid_dir=neural_full_grid,
            make_figures=make_figures,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Execution centrepiece build failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB execution centrepiece builder")
    print(f"  execution analysis: {summary.execution_analysis_dir}")
    print(f"  output directory:   {summary.output_dir}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    print("  interpretation:")
    for claim_id, status in summary.claim_statuses.items():
        print(f"    {claim_id}: {status}")
    if summary.figures_generated:
        print("  figures:            " + ", ".join(summary.figures_generated))
    else:
        print("  figures:            none")
    print("  raw predictions:    not required")
    print("  network calls:      none performed")
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
        print("  predictive metrics:  " + ", ".join(summary.predictive_metric_names))
    if summary.calibration_metric_names:
        print("  calibration metrics: " + ", ".join(summary.calibration_metric_names))
    if summary.metric_names:
        print(f"  all metrics emitted: {', '.join(summary.metric_names)}")
    else:
        print("  metrics emitted:     none")
    print(f"  fixture run:         {'yes' if summary.is_fixture else 'no'}")
    print(f"  runner version:      {summary.runner_version}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    print(f"  artefact validation: {'valid' if summary.validation.is_valid else 'invalid'}")
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
                (result.reason for result in summary.results if result.name == name),
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
            "  supported benchmark sets: " + ", ".join(SUPPORTED_SYSTEM_BENCHMARK_SETS),
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
        sorted(path.name for path in reports_dir.glob("*.md")) if reports_dir.is_dir() else []
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
    print(f"  artefact validation: {'valid' if report.is_valid else 'invalid'}")
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
                print("    robustness:   " + ", ".join(streams.robustness))
    else:
        print("  results.json:     not found")

    predictions_path = resolved_dir / "predictions.csv"
    if predictions_path.is_file():
        try:
            row_count = sum(1 for _ in predictions_path.open("r", encoding="utf-8")) - 1
        except OSError as exc:
            print(f"  predictions.csv row count error: {exc}")
        else:
            print("  prediction rows:  " + (str(max(row_count, 0)) if row_count >= 0 else "0"))
    else:
        print("  prediction rows:  predictions.csv not present")

    calibration_path = resolved_dir / "calibration_bins.csv"
    if calibration_path.is_file():
        try:
            calibration_count = sum(1 for _ in calibration_path.open("r", encoding="utf-8")) - 1
        except OSError as exc:
            print(f"  calibration row count error: {exc}")
        else:
            print("  calibration rows: " + str(max(calibration_count, 0)))
    else:
        print("  calibration rows: calibration_bins.csv not present")

    execution_path = resolved_dir / "execution_sensitivity.csv"
    if execution_path.is_file():
        try:
            execution_count = sum(1 for _ in execution_path.open("r", encoding="utf-8")) - 1
        except OSError as exc:
            print(f"  execution row count error: {exc}")
        else:
            print("  execution rows:   " + str(max(execution_count, 0)))
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
    print(f"  fixture/smoke run:  {'yes' if summary.fixture_or_smoke_run else 'no'}")
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
        print(f"  fixture/smoke flag:   {'yes' if inspection.fixture_or_smoke_run else 'no'}")
    print("  outputs:              not written")
    print("  network calls:        none performed")
    return 0
