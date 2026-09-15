"""FI-2010 experiment command implementations."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


def _audit_fi2010_features_impl(
    *,
    path: Path,
    feature_groups: str | None,
    label_columns: str | None,
    split_column: str | None,
    strict: bool,
    volatility_window: int,
) -> int:
    """Audit FI-2010 microstructure feature construction."""
    from chronoslob.features.microstructure_fi2010 import audit_fi2010_feature_file

    labels = (
        None
        if label_columns is None or not label_columns.strip()
        else [token.strip() for token in label_columns.split(",") if token.strip()]
    )
    try:
        report = audit_fi2010_feature_file(
            path,
            label_columns=labels,
            feature_groups=feature_groups,
            split_column=split_column,
            strict=strict,
            volatility_window=volatility_window,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"FI-2010 feature audit failed: {exc}", file=sys.stderr)
        return 1

    checks = report.get("checks", {})
    print("ChronosLOB FI-2010 microstructure feature audit")
    print(f"  input:               {report.get('input_path')}")
    print(f"  status:              {report.get('status')}")
    print(f"  strict mode:         {'yes' if strict else 'no'}")
    print(f"  unsupported groups:  {len(report.get('unsupported_groups', []))}")
    print(f"  proxy groups:        {len(report.get('proxy_groups', []))}")
    for name in (
        "no_label_columns_used",
        "no_future_horizon_columns_used",
        "rolling_volatility_past_only",
        "snapshot_delta_proxy_no_cross_boundary",
        "train_validation_test_boundaries_respected",
        "row_alignment",
        "missing_column_checks",
    ):
        check = checks.get(name, {})
        status = "pass" if check.get("passed") else "fail"
        print(f"  {name}: {status}")
    if report.get("warnings"):
        print("  warnings:")
        for warning in report["warnings"]:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  outputs:             not written")
    print("  network calls:       none performed")
    return 0 if report.get("status") == "pass" else 1


def _run_fi2010_feature_ablations_impl(
    *,
    config_path: Path | None,
    processed_root: Path | None,
    data_path: Path | None,
    out: Path,
    folds: str | None,
    horizons: str | None,
    seeds: str | None,
    models: str | None,
    feature_groups: str | None,
    ablation_modes: str | None,
    reuse_completed: bool,
    strict: bool,
    smoke_test: bool,
    save_predictions: bool,
    save_heavy_artefacts: bool,
    summary_only: bool,
) -> int:
    """Run the FI-2010 microstructure feature ablation pipeline."""
    from chronoslob.experiments.fi2010_feature_ablations import (
        ABLATION_MODES,
        CLASSICAL_FEATURE_ABLATION_MODELS,
        run_fi2010_feature_ablations,
    )

    try:
        summary = run_fi2010_feature_ablations(
            config_path=config_path,
            processed_root=processed_root,
            data_path=data_path,
            out_dir=out,
            folds=folds,
            horizons=horizons,
            seeds=seeds,
            models=models,
            feature_groups=feature_groups,
            ablation_modes=ablation_modes,
            reuse_completed=reuse_completed,
            strict=strict,
            smoke_test=smoke_test,
            save_predictions=save_predictions,
            save_heavy_artefacts=save_heavy_artefacts,
            summary_only=summary_only,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"FI-2010 feature ablations failed: {exc}", file=sys.stderr)
        print(
            "  supported models: " + ", ".join(CLASSICAL_FEATURE_ABLATION_MODELS),
            file=sys.stderr,
        )
        print(
            "  supported ablation modes: " + ", ".join(ABLATION_MODES),
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB FI-2010 microstructure feature ablations")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  smoke test:          {'yes' if summary.smoke_test else 'no'}")
    print(f"  folds:               {', '.join(summary.folds) or 'none'}")
    print(f"  horizons:            {summary.horizons}")
    print(f"  seeds:               {summary.seeds}")
    print(f"  models:              {', '.join(summary.models)}")
    print(f"  feature groups:      {', '.join(summary.feature_groups)}")
    print(f"  ablation modes:      {', '.join(summary.ablation_modes)}")
    print(f"  planned rows:        {summary.run_count}")
    print(f"  completed rows:      {summary.completed_run_count}")
    print(f"  failed rows:         {summary.failed_run_count}")
    print(f"  summary only:        {'yes' if summary.summary_only else 'no'}")
    print(f"  raw predictions:     {'written' if summary.save_predictions else 'not written'}")
    print(f"  feature matrices:    {'written' if summary.save_heavy_artefacts else 'not written'}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  neural runs:         not run by this classical-first ablation pipeline")
    print("  network calls:       none performed")
    return 0 if summary.failed_run_count == 0 else 1


def _build_fi2010_ablation_figures_impl(
    *,
    ablations: Path,
    out: Path,
    overwrite: bool,
    allow_smoke_test: bool,
) -> int:
    """Build FI-2010 feature-ablation figures from stored artefacts."""
    from chronoslob.analysis.fi2010_ablation_figures import (
        build_fi2010_ablation_figures,
    )

    try:
        summary = build_fi2010_ablation_figures(
            ablation_dir=ablations,
            out_dir=out,
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
        print(f"FI-2010 ablation figure generation failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 feature-ablation figure builder")
    print(f"  ablation dir:        {summary.ablation_dir}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  smoke test:          {'yes' if summary.smoke_test else 'no'}")
    print(f"  manifest:            {summary.manifest_path}")
    print(f"  completed figures:   {len(summary.completed_figures)}")
    for figure_id in summary.completed_figures:
        print(f"    - {figure_id}")
    print(f"  skipped figures:     {len(summary.skipped_figures)}")
    for figure_id in summary.skipped_figures:
        print(f"    - {figure_id}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  network calls:       none performed")
    return 0


def _inspect_fi2010_multifold_impl(
    *,
    config_path: Path,
    extracted_root: Path,
    processed_root: Path | None,
    folds: list[int] | None,
) -> int:
    """Report configured folds and which expected files are present."""
    from chronoslob.experiments.fi2010_multifold import (
        inspect_multifold_files,
        load_multifold_config,
    )

    try:
        config = load_multifold_config(config_path)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"Failed to load multi-fold config: {exc}", file=sys.stderr)
        return 1

    resolved_processed = (
        Path(processed_root)
        if processed_root is not None
        else Path(config.preparation.processed_output_root_placeholder)
    )

    try:
        plans = inspect_multifold_files(
            config,
            extracted_root=Path(extracted_root),
            processed_root=resolved_processed,
            folds=folds,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"Multi-fold inspection failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 multi-fold inspection")
    print(f"  config:              {config_path}")
    print(f"  study name:          {config.study_name}")
    print(f"  extracted root:      {extracted_root}")
    print(f"  processed root:      {resolved_processed}")
    print(f"  configured folds:    {list(config.folds)}")
    print(f"  requested folds:     {[plan.fold for plan in plans]}")
    print("  fold file status:")
    all_ready = True
    for plan in plans:
        ready = "ready" if plan.is_ready else "missing"
        if not plan.is_ready:
            all_ready = False
        print(f"    fold {plan.fold}: {ready}")
        train_state = "present" if plan.train_present else "MISSING"
        test_state = "present" if plan.test_present else "MISSING"
        print(f"      train: {train_state} ({plan.train_path})")
        print(f"      test:  {test_state} ({plan.test_path})")
        print(f"      combined output (planned): {plan.combined_output_path}")
    print("  outputs:             not written (inspection only)")
    print("  network calls:       none performed")
    return 0 if all_ready else 1


def _prepare_fi2010_multifold_impl(
    *,
    config_path: Path,
    extracted_root: Path,
    processed_root: Path | None,
    out: Path,
    folds: list[int] | None,
    overwrite: bool,
) -> int:
    """Prepare combined CSVs and manifests for the requested folds."""
    from chronoslob.experiments.fi2010_multifold import (
        load_multifold_config,
        prepare_multifold,
    )

    try:
        config = load_multifold_config(config_path)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"Failed to load multi-fold config: {exc}", file=sys.stderr)
        return 1

    resolved_processed = (
        Path(processed_root)
        if processed_root is not None
        else Path(config.preparation.processed_output_root_placeholder)
    )

    try:
        result = prepare_multifold(
            config=config,
            config_source_path=Path(config_path),
            extracted_root=Path(extracted_root),
            processed_root=resolved_processed,
            output_dir=Path(out),
            folds=folds,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except NotADirectoryError as exc:
        print(f"Path is not a directory: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"FI-2010 multi-fold preparation failed: {exc}", file=sys.stderr)
        return 1

    summary = result.summary
    print("ChronosLOB FI-2010 multi-fold preparation")
    print(f"  config:              {config_path}")
    print(f"  study name:          {summary.study_name}")
    print(f"  extracted root:      {summary.extracted_dataset_root}")
    print(f"  processed root:      {summary.processed_output_root}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  configured folds:    {summary.folds_configured}")
    print(f"  prepared folds:      {summary.folds_prepared}")
    if summary.folds_skipped:
        print(f"  skipped folds:       {summary.folds_skipped}")
    else:
        print("  skipped folds:       none")
    for manifest in result.fold_manifests:
        train_count = manifest.split_counts.get(config.train_value, 0)
        test_count = manifest.split_counts.get(config.test_value, 0)
        print(
            f"    fold {manifest.fold}: rows={manifest.combined_row_count} "
            f"train={train_count} test={test_count} "
            f"combined={manifest.combined_csv_absolute_path}"
        )
    print(f"  summary path:        {Path(summary.output_dir) / 'summary.json'}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  predictions:         not written (preparation only)")
    print("  model results:       not written (preparation only)")
    print("  network calls:       none performed")
    return 0


def _run_fi2010_multifold_classical_impl(
    *,
    config_path: Path,
    processed_root: Path | None,
    out: Path,
    models: Sequence[str] | None,
    folds: list[int] | None,
    overwrite: bool,
) -> int:
    """Run the FI-2010 multi-fold classical benchmark layer."""
    from chronoslob.experiments.fi2010_multifold_runner import (
        CLASSICAL_MULTIFOLD_MODELS,
        run_fi2010_multifold_classical,
    )

    try:
        summary = run_fi2010_multifold_classical(
            config_path=Path(config_path),
            processed_root=Path(processed_root) if processed_root is not None else None,
            out_dir=Path(out),
            models=models,
            folds=folds,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"FI-2010 multi-fold classical run failed: {exc}", file=sys.stderr)
        print(
            "  supported classical models: " + ", ".join(CLASSICAL_MULTIFOLD_MODELS),
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB FI-2010 multi-fold classical runner")
    print(f"  study name:          {summary.study_name}")
    print(f"  dataset name:        {summary.dataset_name}")
    print(f"  task name:           {summary.task_name}")
    print(f"  horizon:             {summary.target_horizon}")
    print(f"  config:              {summary.config_path}")
    print(f"  processed root:      {summary.processed_root}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  folds:               {summary.folds_completed}")
    print(f"  fold count:          {summary.fold_count}")
    print(f"  models:              {', '.join(summary.models_requested)}")
    print(f"  model count:         {summary.model_count}")
    print(f"  seeds:               {summary.seeds}")
    print(f"  result rows:         {summary.result_rows}")
    print(f"  model failures:      {summary.failure_count}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    print("  full predictions:    not written")
    print("  network calls:       none performed")
    return 0 if summary.failure_count == 0 else 1


def _run_fi2010_brutal_ablations_impl(
    *,
    config_path: Path,
    neural_config_path: Path | None,
    processed_root: Path | None,
    classical_dir: Path | None,
    neural_dir: Path | None,
    out: Path,
    families: str | None,
    folds: str | None,
    models: str | None,
    neural_lookbacks: str | None,
    max_epochs: int,
    overwrite: bool,
    dry_run: bool,
) -> int:
    """Run the FI-2010 brutal ablation layer and report a concise summary."""
    from chronoslob.experiments.fi2010_brutal_ablations import (
        ABLATION_FAMILIES,
        run_fi2010_brutal_ablations,
    )

    try:
        summary = run_fi2010_brutal_ablations(
            config_path=config_path,
            neural_config_path=neural_config_path,
            processed_root=processed_root,
            classical_dir=classical_dir,
            neural_dir=neural_dir,
            out_dir=out,
            families=families,
            folds=folds,
            models=models,
            neural_lookbacks=neural_lookbacks,
            max_epochs=max_epochs,
            overwrite=overwrite,
            dry_run=dry_run,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"FI-2010 brutal ablations failed: {exc}", file=sys.stderr)
        print(
            "  supported families: " + ", ".join(ABLATION_FAMILIES),
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB FI-2010 brutal ablations")
    print(f"  config:              {summary.config_path}")
    print(f"  neural config:       {summary.neural_config_path}")
    print(f"  processed root:      {summary.processed_root}")
    print(f"  classical dir:       {summary.classical_dir}")
    print(f"  neural dir:          {summary.neural_dir}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  families requested:  {', '.join(summary.families_requested)}")
    if summary.dry_run:
        print("  mode:                dry-run (no artefacts written)")
        print(f"  folds planned:       {', '.join(summary.folds) or 'none'}")
        print(f"  fit models:          {', '.join(summary.fit_models)}")
        print("  network calls:       none performed")
        return 0
    print(f"  families run:        {', '.join(summary.families_run) or 'none'}")
    print(f"  families skipped:    {', '.join(summary.families_skipped) or 'none'}")
    print(f"  folds:               {', '.join(summary.folds) or 'none'}")
    print(f"  fit models:          {', '.join(summary.fit_models)}")
    print(f"  feature groups:      {', '.join(summary.feature_groups) or 'none'}")
    print(f"  result rows:         {summary.result_row_count}")
    print(f"  ok rows:             {summary.ok_row_count}")
    print(f"  skipped rows:        {summary.skipped_count}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  full predictions:    not written")
    print("  checkpoints:         not written")
    print("  network calls:       none performed")
    return 0


def _run_fi2010_execution_v2_impl(
    *,
    classical_dir: Path | None,
    neural_dir: Path | None,
    ablations_dir: Path | None,
    out: Path,
    models: str | None,
    cost_bps: str | None,
    latency_steps: str | None,
    confidence_thresholds: str | None,
    overwrite: bool,
) -> int:
    """Build FI-2010 execution-aware v2 proxy diagnostics and report a summary."""
    from chronoslob.experiments.execution_v2 import run_fi2010_execution_v2

    try:
        summary = run_fi2010_execution_v2(
            classical_dir=classical_dir,
            neural_dir=neural_dir,
            ablations_dir=ablations_dir,
            out_dir=out,
            models=models,
            cost_bps=cost_bps,
            latency_steps=latency_steps,
            confidence_thresholds=confidence_thresholds,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"FI-2010 execution v2 failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 execution-aware evaluation v2")
    print(f"  classical dir:       {summary.classical_dir}")
    print(f"  neural dir:          {summary.neural_dir}")
    print(f"  ablations dir:       {summary.ablations_dir}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  classical models:    {', '.join(summary.classical_models) or 'none'}")
    print(f"  neural models:       {', '.join(summary.neural_models) or 'none'}")
    print(f"  folds:               {', '.join(summary.folds) or 'none'}")
    print(f"  result rows:         {summary.result_row_count}")
    print(f"  ok rows:             {summary.ok_row_count}")
    print(f"  skipped rows:        {summary.skipped_row_count}")
    print(f"  diagnostics:         {', '.join(summary.diagnostics_produced) or 'none'}")
    print(f"  skipped diagnostics: {', '.join(summary.diagnostics_skipped) or 'none'}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  metrics:             proxy diagnostics; tradability is not estimated")
    print("  full predictions:    not required")
    print("  checkpoints:         not required")
    print("  network calls:       none performed")
    return 0


def _build_fi2010_execution_v3_impl(
    *,
    neural_full_grid: Path,
    feature_ablations: Path | None,
    out: Path,
    models: str | None,
    horizons: str | None,
    folds: str | None,
    seeds: str | None,
    confidence_thresholds: str | None,
    fee_bps: str | None,
    spread_multipliers: str | None,
    latency_steps: str | None,
    fill_assumptions: str | None,
    allow_smoke_test: bool,
    strict: bool,
    overwrite: bool,
) -> int:
    """Build FI-2010 execution-aware proxy diagnostic v3 from full-grid artefacts."""
    from chronoslob.analysis.execution_v3 import build_fi2010_execution_v3

    try:
        summary = build_fi2010_execution_v3(
            neural_full_grid_dir=Path(neural_full_grid),
            feature_ablation_dir=(
                Path(feature_ablations) if feature_ablations is not None else None
            ),
            out_dir=Path(out),
            models=models,
            horizons=horizons,
            folds=folds,
            seeds=seeds,
            confidence_thresholds=confidence_thresholds,
            fee_bps=fee_bps,
            spread_multipliers=spread_multipliers,
            latency_steps=latency_steps,
            fill_assumptions=fill_assumptions,
            allow_smoke_test=allow_smoke_test,
            strict=strict,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"FI-2010 execution v3 failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 execution-aware proxy diagnostic v3")
    print(f"  neural full grid:    {summary.neural_full_grid_dir}")
    if summary.feature_ablation_dir is not None:
        print(f"  feature ablations:   {summary.feature_ablation_dir}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  manifest:            {summary.manifest_path}")
    print(f"  summary:             {summary.summary_path}")
    print(f"  prediction rows:     {summary.prediction_row_count}")
    print(f"  run groups:          {summary.run_group_count}")
    print(f"  payoff mode:         {summary.payoff_mode}")
    print(f"  cost mode:           {summary.cost_mode}")
    print(f"  smoke test:          {'yes' if summary.smoke_test else 'no'}")
    print(f"  strict mode:         {'yes' if summary.strict else 'no'}")
    print(f"  diagnostics:         {', '.join(summary.diagnostics_produced) or 'none'}")
    print(f"  skipped diagnostics: {', '.join(summary.diagnostics_skipped) or 'none'}")
    print("  artefacts written:")
    for key, relative_path in summary.output_files.items():
        print(f"    {key}: {relative_path}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    else:
        print("  warnings:            none")
    print("  interpretation:      offline execution-aware proxy diagnostic only")
    print("  live trading:        not implemented")
    print("  network calls:       none performed")
    return 0
