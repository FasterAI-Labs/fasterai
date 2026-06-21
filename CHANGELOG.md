# Release notes

<!-- do not remove -->

## 0.3.3

### Bug Fixes
- `PruneCallback` now accepts a per-layer `dict` for `pruning_ratio` (matching `Pruner`); previously raised `TypeError: '>' not supported between instances of 'dict' and 'int'` (#39)
- Sensitivity analysis pruning mode now measures each layer by applying its exact per-layer target `{name: level}` — the same operation `Pruner`/`PruneCallback` perform — so the reported Δ faithfully predicts a real prune. Fixes residual/skip-coupled layers (ResNet stem conv, block `conv2`, `downsample`) being falsely ranked "Most Robust" with Δ=0 because they could not be pruned in isolation (#40)

### Enhancements
- Sensitivity analysis: coupled layers now share a `group_id` (and Δ); layers that cannot be pruned independently (output `Linear`, attention) are marked `prunable=False` and surfaced separately rather than ranked; `analyze(layer_types=...)` restricts the analysis to chosen module types (accepts a single type or a tuple); internal `Pruner` notices are silenced during analysis

### Metadata
- Corrected `requires-python` to `>=3.10` (the codebase uses `dataclass(slots=True)`)

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

