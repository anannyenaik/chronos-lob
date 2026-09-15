# Synthetic FI-2010-style Fixtures

These fixtures are **synthetic test data**. They were hand-crafted to exercise
the FI-2010 loader and validation helpers. They are **not** derived from, and
are **not** equivalent to, the real FI-2010 benchmark.

Specifically:

- numeric values are made-up positive numbers chosen for readability;
- timestamps are timezone-aware ISO-8601 strings;
- the `split` column carries `train` / `test` values for testing split
  detection only;
- labels are integer class identifiers used purely for shape testing.

Do not use these fixtures to make any benchmark, statistical or research
claims about FI-2010. Real benchmark data must be obtained, stored and
managed by the user outside the repository.
