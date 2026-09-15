"""Synthetic smoke-test command implementations."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .data import _print_synthetic_fixture_warning


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
    print("  Scope: supervised architecture inspection only.")
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
        "(synthetic smoke-test metric)"
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
    print("  Self-supervised pretraining over field-wise tokenised market microstructure.")
    print("  Scope: self-supervised architecture inspection only.")
    print(f"  enabled objectives:       {list(config.enabled_objectives())}")
    print(f"  masked fields:            {list(config.masked_fields)}")
    print(f"  next-predicted fields:    {list(config.next_fields)}")
    print(f"  ignore_index:             {config.ignore_index}")
    print(f"  contrastive enabled:      {config.enable_contrastive_loss}")
    print("  masking config:")
    print(f"    mask_probability:        {config.masking.mask_probability}")
    print(f"    mask_token_probability:  {config.masking.mask_token_probability}")
    print(f"    random_token_probability: {config.masking.random_token_probability}")
    print(f"    keep_token_probability:  {config.masking.keep_token_probability}")
    print(f"    force_at_least_one_mask: {config.masking.force_at_least_one_mask}")
    print("  loss weights:")
    for name, weight in dict(config.loss_weights).items():
        print(f"    {name}: {weight}")
    print("  transformer backbone:")
    print(f"    model_dim:              {config.transformer.model_dim}")
    print(f"    num_heads:              {config.transformer.num_heads}")
    print(f"    num_layers:             {config.transformer.num_layers}")
    print(f"    max_sequence_length:    {config.transformer.max_sequence_length}")
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
        "Synthetic SSL smoke test; losses describe implementation behaviour and "
        "do not represent market signal or benchmark performance."
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
        "(synthetic smoke-test metric)"
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
    print("  Supervised fine-tuning heads over a shared field-wise token transformer backbone.")
    print(
        "  No calibration, confidence filtering, execution simulation, "
        "backtesting or performance estimation."
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
        "Synthetic supervised smoke test; losses and accuracies describe "
        "implementation behaviour and do not represent benchmark performance."
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
    print("  benchmark interpretation:  synthetic smoke test only")
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
        "Synthetic calibration smoke test; metrics describe implementation behaviour "
        "and do not represent benchmark or trading performance."
    )
    print(f"  synthetic examples:      {result['n_examples']}")
    print(f"  calibration examples:    {result['calibration_examples']}")
    print(f"  evaluation examples:     {result['evaluation_examples']}")
    print(f"  number of classes:       {result['num_classes']}")
    print(f"  seed:                    {result['seed']}")
    print(f"  fitted temperature:      {result['fitted_temperature']:.6f}")
    print("  pre-calibration metrics:")
    pre = cast(Mapping[str, Any], result["pre_calibration"])
    print(f"    nll={pre['nll']:.6f} ece={pre['ece']:.6f} brier={pre['brier_score']:.6f}")
    print("  post-calibration metrics:")
    post = cast(Mapping[str, Any], result["post_calibration"])
    print(f"    nll={post['nll']:.6f} ece={post['ece']:.6f} brier={post['brier_score']:.6f}")
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
    print("  statement:          simulation diagnostic; tradability is not estimated")
    return 0


def _run_execution_validation_smoke_impl(
    *,
    n_signals: int = 24,
    seed: int = 42,
) -> int:
    """Run a deterministic synthetic execution-validation smoke test."""
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
        "Synthetic execution-validation smoke test; outputs describe implementation "
        "behaviour and do not represent benchmark or live performance."
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
    regime_summary_counts = cast(Mapping[str, int], result["regime_summary_counts"])
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
    print(f"  ablation comparisons:     {result['ablation_comparisons_count']}")
    print(f"  sensitivity curves:       {result['sensitivity_curves_produced']}")
    print("  example metric summaries:")
    example_rows = cast(Sequence[Mapping[str, Any]], result["example_summary_rows"])
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
        "  WARNING: synthetic analysis smoke test; outputs describe implementation "
        "behaviour and require real experiment records for empirical analysis."
    )
    print("  outputs:                  not written (smoke command)")
    print("  network calls:            none performed")
    return 0
