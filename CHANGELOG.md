# Release notes

<!-- do not remove -->

## 0.3.0

### New Features
- Migrated to nbdev3 with `pyproject.toml` (PEP 621) replacing `settings.ini`
- Renamed `analysis` module to `analyze` for verb-based naming consistency
- Agnostic `Schedule` system: schedules return progress (0→1), callbacks multiply by target
- Per-layer sparsity targets via dict for `SparsifyCallback` and `Pruner`
- Sensitivity analysis module (`fasterai.analyze.sensitivity`)

### Enhancements
- Updated CI workflows to nbdev3 actions (`quarto-ghp3`, `nbdev3-ci`)
- Expanded PyPI keywords and classifiers
- Fixed repository URLs and metadata for FasterAI-Labs organization

