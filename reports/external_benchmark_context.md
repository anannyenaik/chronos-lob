# External Benchmark Context Note

External comparison is necessary because FI-2010 results in the literature are
not automatically comparable. Dataset variant, auction inclusion,
normalisation, folds, horizon, label mapping, split semantics and metric
definition can all change the meaning of a reported score.

Added:

- `docs/FI2010_EXTERNAL_BENCHMARKS.md`, a concise public protocol-comparison
  document.
- `experiments/fi2010_external_context/`, a small structured context artefact
  with JSON, CSV and notes.

Not claimed:

- no external-ranking or neural-superiority claim;
- no profitability, live tradability or deployment-quality claim;
- no foundation-model claim;
- no SSL result;
- no unverified external numeric paper metrics.

Future work that would make direct comparison stronger:

- verify external paper metrics from primary sources and record exact table,
  fold, horizon and metric definitions;
- reproduce selected external protocols locally when licences and compute
  budget allow;
- complete multi-seed and multi-lookback neural evidence before any stronger
  neural comparison;
- add direct comparison rows only when the local and external protocol
  dimensions match.
