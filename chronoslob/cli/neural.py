"""Neural benchmark command implementations."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


def _inspect_fi2010_neural_plan_impl(
    *,
    config_path: Path,
    folds: list[int] | None,
    models: Sequence[str] | None,
) -> int:
    """Inspect the serious FI-2010 neural benchmark plan without training."""
    from chronoslob.experiments.neural_benchmarking import (
        expected_lightweight_artefacts,
        generate_neural_run_plan,
        load_neural_benchmark_config,
        resolve_neural_device,
    )

    try:
        config = load_neural_benchmark_config(config_path)
        plan = generate_neural_run_plan(config, folds=folds, models=models)
        device = resolve_neural_device(config.device_selection)
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"FI-2010 neural plan inspection failed: {exc}", file=sys.stderr)
        return 1

    selected_folds = list(dict.fromkeys(item.fold_id for item in plan))
    selected_models = list(dict.fromkeys(item.model_name for item in plan))
    selected_lookbacks = list(dict.fromkeys(item.lookback for item in plan))
    artefacts = expected_lightweight_artefacts(config)

    print("ChronosLOB FI-2010 neural benchmark plan")
    print(f"  config:                 {config_path}")
    print(f"  study name:             {config.study_name}")
    print(f"  mode:                   {config.mode}")
    print(f"  smoke mode:             {'yes' if config.is_smoke_mode else 'no'}")
    print(f"  benchmark mode:         {'yes' if config.is_benchmark_mode else 'no'}")
    print(f"  planned runs:           {len(plan)}")
    print(f"  folds:                  {selected_folds}")
    print(f"  seeds:                  {list(config.seeds)}")
    print(f"  models:                 {selected_models}")
    print(f"  lookbacks:              {selected_lookbacks}")
    print(f"  target horizon:         {config.target.horizon}")
    print(f"  validation metric:      {config.validation_metric}")
    print(f"  max epochs:             {config.training.max_epochs}")
    print(f"  early stopping:         {config.training.early_stopping_metric}")
    print(f"  early stopping patience: {config.training.early_stopping_patience}")
    print(f"  device policy:          {device.requested}")
    print(f"  resolved device:        {device.resolved}")
    print(f"  cuda available:         {'yes' if device.cuda_available else 'no'}")
    print(f"  output root:            {config.artefacts.output_root}")
    print(f"  checkpoint root:        {config.artefacts.checkpoint_root}")
    print("  expected artefacts:")
    for name, path in artefacts.items():
        print(f"    {name}: {path}")
    print(
        "  full predictions:       "
        f"{'written' if config.artefacts.write_full_predictions_by_default else 'not written'}"
    )
    print(
        "  checkpoints by default: "
        f"{'written' if config.artefacts.write_checkpoints_by_default else 'not written'}"
    )
    print("  training:               not run")
    print("  outputs:                not written (inspection only)")
    print("  network calls:          none performed")
    return 0


def _run_fi2010_neural_benchmark_impl(
    *,
    config_path: Path,
    processed_root: Path,
    out: Path,
    folds: Sequence[str] | None,
    models: Sequence[str] | None,
    seeds: Sequence[int] | None,
    lookbacks: Sequence[int] | None,
    max_epochs: int,
    overwrite: bool,
    fail_fast: bool,
    write_full_predictions: bool,
    write_checkpoints: bool,
    allow_full_benchmark: bool,
) -> int:
    """Run selected FI-2010 supervised neural benchmark configurations."""
    from chronoslob.experiments.fi2010_neural_runner import (
        run_fi2010_neural_benchmark,
    )
    from chronoslob.experiments.neural_benchmarking import (
        SUPPORTED_NEURAL_BENCHMARK_MODELS,
    )

    try:
        summary = run_fi2010_neural_benchmark(
            config_path=Path(config_path),
            processed_root=Path(processed_root),
            out_dir=Path(out),
            folds=folds,
            models=models,
            seeds=seeds,
            lookbacks=lookbacks,
            max_epochs=max_epochs,
            overwrite=overwrite,
            fail_fast=fail_fast,
            write_full_predictions=write_full_predictions,
            write_checkpoints=write_checkpoints,
            allow_full_benchmark=allow_full_benchmark,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"FI-2010 neural benchmark run failed: {exc}", file=sys.stderr)
        print(
            "  supported neural models: " + ", ".join(SUPPORTED_NEURAL_BENCHMARK_MODELS),
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB FI-2010 neural benchmark runner")
    print(f"  study name:          {summary.study_name}")
    print(f"  dataset name:        {summary.dataset_name}")
    print(f"  task name:           {summary.task_name}")
    print(f"  horizon:             {summary.target_horizon}")
    print(f"  config:              {summary.config_path}")
    print(f"  processed root:      {summary.processed_root}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  execution mode:      {summary.execution_mode}")
    print(f"  benchmark-level:     {'yes' if summary.full_benchmark_grid else 'no'}")
    print(f"  folds:               {summary.folds_requested}")
    print(f"  seeds:               {summary.seeds}")
    print(f"  models:              {', '.join(summary.models_requested)}")
    print(f"  lookbacks:           {summary.lookbacks}")
    print(f"  max epochs:          {summary.max_epochs}")
    print(f"  planned runs:        {summary.run_count}")
    print(f"  completed runs:      {summary.completed_run_count}")
    print(f"  model failures:      {summary.failure_count}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    print(
        f"  full predictions:    {'written' if summary.full_predictions_written else 'not written'}"
    )
    print(f"  checkpoints:         {'written' if summary.checkpoints_written else 'not written'}")
    print("  network calls:       none performed")
    return 0 if summary.failure_count == 0 else 1


def _run_fi2010_ssl_neural_benchmark_impl(
    *,
    config_path: Path,
    processed_root: Path,
    out: Path,
    folds: Sequence[str] | None,
    seeds: Sequence[int] | None,
    lookbacks: Sequence[int] | None,
    objective: str,
    mask_probability: float,
    next_field_bucket_count: int,
    pretrain_epochs: int,
    max_epochs: int,
    batch_size: int,
    device: str,
    overwrite: bool,
    fail_fast: bool,
    write_full_predictions: bool,
) -> int:
    """Run the FI-2010 SSL pretraining and fine-tuning benchmark."""
    from chronoslob.experiments.fi2010_ssl_runner import (
        SSL_OBJECTIVE_CHOICES,
        run_fi2010_ssl_neural_benchmark,
    )

    try:
        summary = run_fi2010_ssl_neural_benchmark(
            config_path=Path(config_path),
            processed_root=Path(processed_root),
            out_dir=Path(out),
            folds=folds,
            seeds=seeds,
            lookbacks=lookbacks,
            objective=objective,
            mask_probability=mask_probability,
            next_field_bucket_count=next_field_bucket_count,
            pretrain_epochs=pretrain_epochs,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            overwrite=overwrite,
            fail_fast=fail_fast,
            write_full_predictions=write_full_predictions,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"FI-2010 SSL benchmark run failed: {exc}", file=sys.stderr)
        print(
            "  supported SSL objectives: " + ", ".join(SSL_OBJECTIVE_CHOICES),
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB FI-2010 SSL pretraining + fine-tuning runner")
    print(f"  study name:          {summary.study_name}")
    print(f"  dataset name:        {summary.dataset_name}")
    print(f"  task name:           {summary.task_name}")
    print(f"  horizon:             {summary.target_horizon}")
    print(f"  config:              {summary.config_path}")
    print(f"  processed root:      {summary.processed_root}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  execution mode:      {summary.execution_mode}")
    print(f"  objective:           {summary.objective}")
    print(f"  folds:               {summary.folds_requested}")
    print(f"  seeds:               {summary.seeds}")
    print(f"  lookbacks:           {summary.lookbacks}")
    print(f"  pretrain epochs:     {summary.pretrain_epochs}")
    print(f"  fine-tune max epochs:{summary.max_epochs}")
    print(f"  planned runs:        {summary.run_count}")
    print(f"  completed runs:      {summary.completed_run_count}")
    print(f"  run failures:        {summary.failure_count}")
    print(f"  ssl artefacts:       {'written' if summary.ssl_artefacts_written else 'not written'}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    print(
        "  comparison:          ssl_transformer vs supervised_transformer "
        "(identical architecture, folds, horizons, seeds and preprocessing)"
    )
    print("  network calls:       none performed")
    return 0 if summary.failure_count == 0 else 1


def _run_fi2010_neural_full_grid_impl(
    *,
    config_path: Path,
    processed_root: Path,
    out: Path,
    folds: Sequence[str | int] | None,
    horizons: Sequence[int] | None,
    seeds: Sequence[int] | None,
    lookbacks: Sequence[int] | None,
    objectives: Sequence[str] | None,
    pretrain_epochs: int,
    max_epochs: int,
    batch_size: int,
    device: str,
    reuse_completed: bool,
    smoke_test: bool,
) -> int:
    """Run the full FI-2010 supervised-vs-SSL neural evidence grid."""
    from chronoslob.experiments.fi2010_neural_grid import (
        GRID_OBJECTIVE_CHOICES,
        run_fi2010_neural_full_grid,
    )

    try:
        summary = run_fi2010_neural_full_grid(
            config_path=Path(config_path),
            processed_root=Path(processed_root),
            out_dir=Path(out),
            folds=folds,
            horizons=horizons,
            seeds=seeds,
            lookbacks=lookbacks,
            objectives=objectives,
            pretrain_epochs=pretrain_epochs,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            reuse_completed=reuse_completed,
            smoke_test=smoke_test,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"FI-2010 neural full grid failed: {exc}", file=sys.stderr)
        print(
            "  supported objectives: " + ", ".join(GRID_OBJECTIVE_CHOICES),
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB FI-2010 neural full grid runner")
    print(f"  config:              {summary.config_path}")
    print(f"  processed root:      {summary.processed_root}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  execution mode:      {summary.execution_mode}")
    print(f"  smoke test:          {'yes' if summary.smoke_test else 'no'}")
    print(f"  folds:               {summary.folds}")
    print(f"  horizons:            {summary.horizons}")
    print(f"  seeds:               {summary.seeds}")
    print(f"  lookbacks:           {summary.lookbacks}")
    print(f"  objectives:          {', '.join(summary.objectives)}")
    print(f"  pretrain epochs:     {summary.pretrain_epochs}")
    print(f"  max epochs:          {summary.max_epochs}")
    print(f"  batch size:          {summary.batch_size}")
    print(f"  device:              {summary.device}")
    print(f"  planned runs:        {summary.run_count}")
    print(f"  completed runs:      {summary.completed_run_count}")
    print(f"  skipped existing:    {summary.skipped_existing_count}")
    print(f"  failed runs:         {summary.failed_run_count}")
    print(f"  missing pairs:       {summary.missing_pair_count}")
    print(f"  core grid complete:  {'yes' if summary.core_grid_complete else 'no'}")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    print("  network calls:       none performed")
    return 0 if summary.failed_run_count == 0 else 1


def _run_fi2010_neural_proper_training_subset_impl(
    *,
    config_path: Path,
    processed_root: Path,
    out: Path,
    folds: Sequence[str | int] | None,
    horizons: Sequence[int] | None,
    seeds: Sequence[int] | None,
    lookbacks: Sequence[int] | None,
    models: Sequence[str] | None,
    objectives: Sequence[str] | None,
    pretrain_epochs: int,
    max_epochs: int | None,
    patience: int | None,
    batch_size: int | None,
    device: str,
    reuse_completed: bool,
    smoke_test: bool,
) -> int:
    """Run the FI-2010 validation-selected neural benchmark."""
    from chronoslob.experiments.fi2010_neural_proper_training import (
        PROPER_TRAINING_MODEL_CHOICES,
        PROPER_TRAINING_OBJECTIVE_CHOICES,
        run_fi2010_neural_proper_training_subset,
    )

    try:
        summary = run_fi2010_neural_proper_training_subset(
            config_path=Path(config_path),
            processed_root=Path(processed_root),
            out_dir=Path(out),
            folds=folds,
            horizons=horizons,
            seeds=seeds,
            lookbacks=lookbacks,
            models=models,
            objectives=objectives,
            pretrain_epochs=pretrain_epochs,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            reuse_completed=reuse_completed,
            smoke_test=smoke_test,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"FI-2010 neural benchmark failed: {exc}", file=sys.stderr)
        print(
            "  supported objectives: " + ", ".join(PROPER_TRAINING_OBJECTIVE_CHOICES),
            file=sys.stderr,
        )
        print(
            "  supported models: " + ", ".join(PROPER_TRAINING_MODEL_CHOICES),
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB FI-2010 validation-selected neural benchmark")
    print(f"  config:                {summary.config_path}")
    print(f"  processed root:        {summary.processed_root}")
    print(f"  output directory:      {summary.output_dir}")
    print(f"  subset kind:           {summary.subset_kind}")
    print(f"  execution mode:        {summary.execution_mode}")
    print(f"  smoke test:            {'yes' if summary.smoke_test else 'no'}")
    print(f"  folds:                 {summary.folds}")
    print(f"  horizons:              {summary.horizons}")
    print(f"  seeds:                 {summary.seeds}")
    print(f"  lookbacks:             {summary.lookbacks}")
    print(f"  models:                {', '.join(summary.models)}")
    print(f"  objectives:            {', '.join(summary.objectives)}")
    print(f"  pretrain epochs:       {summary.pretrain_epochs}")
    print(f"  max epochs:            {summary.max_epochs}")
    print(f"  early stopping metric: {summary.early_stopping_metric}")
    print(f"  early stopping patience: {summary.early_stopping_patience}")
    print(f"  batch size:            {summary.batch_size}")
    print(f"  device:                {summary.device}")
    print(f"  planned runs:          {summary.run_count}")
    print(f"  completed runs:        {summary.completed_run_count}")
    print(f"  skipped existing:      {summary.skipped_existing_count}")
    print(f"  failed runs:           {summary.failed_run_count}")
    print(f"  missing pairs:         {summary.missing_pair_count}")
    print(f"  target scope complete: {'yes' if summary.target_scope_complete else 'no'}")
    print("  validation-only model selection; best checkpoint restored before test")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    print("  network calls:         none performed")
    return 0 if summary.failed_run_count == 0 else 1


def _run_fi2010_ssl_v2_benchmark_impl(
    *,
    config_path: Path,
    processed_root: Path,
    out: Path,
    baseline_source: Path | None,
    folds: Sequence[str | int] | None,
    horizons: Sequence[int] | None,
    seeds: Sequence[int] | None,
    lookbacks: Sequence[int] | None,
    objectives: Sequence[str] | None,
    pretrain_epochs: int,
    max_epochs: int | None,
    patience: int | None,
    batch_size: int | None,
    mask_probability: float,
    future_bucket_count: int,
    contrastive: bool,
    device: str,
    reuse_completed: bool,
    import_existing_baselines: bool,
    smoke_test: bool,
) -> int:
    """Run the FI-2010 SSL-v2 benchmark."""
    from chronoslob.experiments.fi2010_ssl_v2_benchmark import (
        SSL_V2_OBJECTIVE_CHOICES,
        run_fi2010_ssl_v2_benchmark,
    )

    try:
        summary = run_fi2010_ssl_v2_benchmark(
            config_path=Path(config_path),
            processed_root=Path(processed_root),
            out_dir=Path(out),
            baseline_source_dir=Path(baseline_source) if baseline_source is not None else None,
            folds=folds,
            horizons=horizons,
            seeds=seeds,
            lookbacks=lookbacks,
            objectives=objectives,
            pretrain_epochs=pretrain_epochs,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            mask_probability=mask_probability,
            future_bucket_count=future_bucket_count,
            contrastive=contrastive,
            device=device,
            reuse_completed=reuse_completed,
            import_existing_baselines=import_existing_baselines,
            smoke_test=smoke_test,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"FI-2010 SSL-v2 benchmark failed: {exc}", file=sys.stderr)
        print(
            "  supported objectives: " + ", ".join(SSL_V2_OBJECTIVE_CHOICES),
            file=sys.stderr,
        )
        return 1

    print("ChronosLOB FI-2010 SSL-v2 benchmark runner")
    print(f"  config:                {summary.config_path}")
    print(f"  processed root:        {summary.processed_root}")
    print(f"  output directory:      {summary.output_dir}")
    print(f"  result coverage:       {summary.evidence_level}")
    print(f"  scope label:           {summary.scope_label}")
    print(f"  folds:                 {summary.folds}")
    print(f"  horizons:              {summary.horizons}")
    print(f"  seeds:                 {summary.seeds}")
    print(f"  lookbacks:             {summary.lookbacks}")
    print(f"  objectives:            {', '.join(summary.objectives)}")
    print(f"  pretrain epochs:       {summary.pretrain_epochs}")
    print(f"  max epochs:            {summary.max_epochs}")
    print(f"  early stopping patience: {summary.early_stopping_patience}")
    print(f"  batch size:            {summary.batch_size}")
    print(f"  device:                {summary.device}")
    print(f"  planned runs:          {summary.run_count}")
    print(f"  completed runs:        {summary.completed_run_count}")
    print(f"  imported baselines:    {summary.imported_baseline_count}")
    print(f"  failed runs:           {summary.failed_run_count}")
    print(f"  missing pairs:         {summary.missing_pair_count}")
    print("  validation-only model selection; best checkpoint restored before test")
    print("  artefacts written:")
    for key, relative_path in summary.artefacts.items():
        print(f"    {key}: {relative_path}")
    if summary.warnings:
        print("  warnings:")
        for warning in summary.warnings:
            print(f"    - {warning}")
    print("  network calls:         none performed")
    return 0 if summary.failed_run_count == 0 else 1


def _build_fi2010_figures_impl(
    *,
    neural_full_grid: Path,
    out: Path,
    execution_v3: Path | None,
    models: Sequence[str] | None,
    horizons: Sequence[int] | None,
    folds: Sequence[str | int] | None,
    seeds: Sequence[int] | None,
    overwrite: bool,
    allow_smoke_test: bool,
    strict: bool,
) -> int:
    """Build FI-2010 neural full-grid diagnostic figures from artefacts."""
    from chronoslob.analysis.fi2010_figures import build_fi2010_neural_figures

    try:
        summary = build_fi2010_neural_figures(
            neural_full_grid_dir=Path(neural_full_grid),
            out_dir=Path(out),
            execution_v3_dir=Path(execution_v3) if execution_v3 is not None else None,
            models=models,
            horizons=horizons,
            folds=folds,
            seeds=seeds,
            overwrite=overwrite,
            allow_smoke_test=allow_smoke_test,
            strict=strict,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"FI-2010 figure generation failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB FI-2010 neural figure builder")
    print(f"  neural full grid:    {summary.neural_full_grid_dir}")
    print(f"  output directory:    {summary.output_dir}")
    print(f"  smoke test:          {'yes' if summary.smoke_test else 'no'}")
    print(f"  manifest:            {summary.manifest_path}")
    print(f"  label audit:         {summary.label_mapping_audit_path}")
    print(f"  best selection:      {summary.best_model_selection_path}")
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
