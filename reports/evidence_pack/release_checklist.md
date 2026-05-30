# Release Checklist

## Git Hygiene

- [ ] Review `git status --short` before release.
- [ ] If the worktree contains unrelated or untracked files, they must be reviewed before release. Do not automatically delete or stage them.

## Tests

- [ ] Run `python -m pytest`.

## Lint And Types

- [ ] Run `python -m ruff check .`.
- [ ] Run `python -m mypy chronoslob`.

## Project Audit

- [ ] Run `python -m chronoslob.cli doctor`.
- [ ] Run `python -m chronoslob.cli inspect-release-readiness`.
- [ ] Run `python -m chronoslob.cli run-project-audit --strict`.

## Artefact Hashes

- [ ] Confirm input hashes are present where artefacts record source files.
- [ ] Investigate genuinely stale rows in artefact_inventory.csv; archived_valid, optional_missing and obsolete_superseded rows are expected and do not block release.

## Smoke And Real Separation

- [ ] Confirm smoke_test_only rows are not cited as empirical evidence.
- [ ] Use `--allow-smoke-test` only for diagnostics packs.

## Public Snapshot

- [ ] Review `readme_result_snapshot.md` before copying any result text.

## Docs And Limits

- [ ] Docs updated.
- [ ] Limitations updated.
- [ ] Unsupported claims removed.

## Public Bullets

- [ ] Check conservative and stronger public bullet files.

## Paper Boundary

- [ ] Manual paper not yet written.

## Blocked Claims

- [ ] No live-trading or profitability claims.
