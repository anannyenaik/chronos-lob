# AGENTS.md

Permanent instructions for Codex agents and other automated contributors working on
ChronosLOB.

## Project Identity

ChronosLOB is a serious ML research-engineering project:

**ChronosLOB: Self-Supervised Market Microstructure Modelling for Execution-Aware
Alpha Discovery**

Recruiter-facing title:

**Self-Supervised Market Microstructure Model with Execution-Aware Alpha Validation**

The project is a reproducible platform for limit order book representation learning,
short-horizon market-state forecasting and execution-aware validation. Treat it as a
research platform, not as a trading product.

## Project Goal

The central research question is whether self-supervised representations of limit
order book dynamics improve short-horizon market-state forecasts, and whether those
forecasts remain useful under execution-aware validation.

Prediction and trading are separate problems. Strong benchmark accuracy is not the
same thing as tradable alpha.

## Vocabulary To Use

Use careful, research-oriented language:

- market microstructure
- limit order book dynamics
- self-supervised representation learning
- short-horizon market-state forecasting
- execution-aware validation
- calibration
- leakage-safe labels
- temporal splits
- regime shift
- cost-adjusted signal quality
- forecasting-versus-tradability gap
- simplified research simulation
- reproducible experiment artefact

## Vocabulary To Avoid

Do not use promotional or misleading language such as:

- AI trading bot
- guaranteed alpha
- beats the market
- profitable strategy
- high Sharpe strategy
- deployable trading system
- money-making signal

Do not market this project as a trading bot.

## Coding Standards

- Prefer explicit names over clever abstractions.
- Use type hints where practical.
- Keep modules small, testable and easy to review.
- Use `pathlib.Path` for filesystem paths.
- Avoid runtime side effects at import time.
- Avoid hidden network calls.
- Do not silently swallow errors in data pipelines.
- Use deterministic behaviour where practical.
- Keep dependency additions conservative and justified.
- Do not add heavy ML dependencies until the relevant modelling phase.
- Keep core logic in the package, not in notebooks.

## Testing Standards

- Every future module should have tests.
- Tests must be meaningful, fast and deterministic.
- Prefer tests that validate behaviour and invariants over tests that only import code.
- Add leakage tests for any feature, label, split or dataset logic.
- Add regression tests for bug fixes.
- Do not use random splits for financial time-series experiments unless a test fixture
  explicitly requires it.

## Data Leakage Rules

- No look-ahead leakage is allowed.
- Features may only use information available at or before timestamp `t`.
- Labels may use future information after timestamp `t`, but labels must never leak
  into features, splits, normalisation statistics or model inputs.
- Temporal train/validation/test splitting is the default for financial data.
- Fit transforms only on the training partition unless a test explicitly demonstrates
  another safe design.
- Document every future data assumption.
- All future data processing must be auditable.

## Experiment Reproducibility Rules

- All future experiments must be reproducible from configuration files.
- Do not report performance without a reproducible experiment artefact.
- Store experiment inputs, config versions, code versions, random seeds and output
  paths.
- Do not invent results.
- Do not create placeholder metrics, fake plots or manually invented result tables.
- Any reported result must be traceable to a reproducible run.

## Financial Modelling Cautions

- Do not treat benchmark accuracy as tradable alpha.
- All backtests are simplified research simulations unless proven otherwise.
- Costs, fees, queue position, latency, market impact, partial fills and venue rules
  require explicit assumptions.
- Crypto market data may be useful for engineering demonstrations, but do not
  overclaim that it is directly equivalent to equities.
- Public data only unless the user later configures private or licensed data.
- The repository is for research and education, not financial advice.

## Documentation Rules

- Use UK English in prose.
- Keep documentation professional, concise and honest.
- State limitations wherever relevant.
- Distinguish implemented functionality from planned work.
- Do not claim that models, loaders, labels, backtests or results exist before they
  are implemented and verified.
- Reports must be generated from reproducible experiment outputs.

## Dependency Rules

- Prefer the standard library when it is sufficient.
- Add dependencies only when they reduce real implementation risk or complexity.
- Do not add PyTorch, JAX, TensorFlow or other heavy ML frameworks until a modelling
  phase requires them.
- Do not add services that make hidden network calls.
- Keep development tooling conventional: pytest, ruff, mypy and pre-commit are
  acceptable.

## How To Report Changes

When finishing a task, report:

1. What changed.
2. Files created or modified.
3. Commands run and whether they passed.
4. Any assumptions made.
5. Follow-up items or known limitations.

## What Not To Do

- Do not invent results.
- Do not report performance without a reproducible experiment artefact.
- Do not let labels leak into features.
- Do not use random splits for financial time-series experiments unless a test
  fixture explicitly requires it.
- Do not market this as a trading bot.
- Do not create fake result files, fake plots or fake notebooks.
- Do not download data unless explicitly requested.
- Do not include secrets, API keys or private data.
- Do not implement FI-2010 loading, Binance ingestion, features, labels, models or
  backtesting before the relevant phase.
