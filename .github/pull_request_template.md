## Summary

- 

## Research And Leakage Checklist

- [ ] No invented results, placeholder metrics or fake plots.
- [ ] No labels leak into features, splits, normalisation or model inputs.
- [ ] Financial time-series splits are temporal unless this is an explicit test fixture.
- [ ] Data assumptions are documented.
- [ ] Limitations are updated where relevant.
- [ ] Tests cover the changed behaviour.

## Verification

- [ ] `python -m pytest`
- [ ] `python -m ruff check .`
- [ ] `python -m mypy chronoslob`
