# Release notes

<!-- do not remove -->

## Unreleased

### New Features
- `Quantizer(backend='torchao', weight_bits={layer: 8|16}, act_bits=16)` applies a per-layer width over the `nn.Linear` layers torchao rewrites, so part of a Linear-heavy model can be held in floating point. Naming every layer `8` produces the same artifact, byte for byte, as the uniform `method='int8_weight_only'` recipe

### Breaking Changes
- A per-layer `weight_bits` dict naming a layer the backend does not quantize now raises `ValueError` instead of being silently ignored. On the legacy backends the names it accepts are the module types their default qconfig mapping rewrites (read from torch, so it tracks the installed version) plus any module containing one; `nn.Embedding`, `nn.EmbeddingBag`, `nn.LSTM`, `nn.GRU` and `nn.RNN` are not among them — the default flow leaves them in floating point. On `backend='torchao'` the names it accepts are the `nn.Linear` layers torchao itself selects, which excludes `MultiheadAttention.out_proj`
- A per-layer `weight_bits` dict that would leave *every* layer the backend rewrites in floating point now raises `ValueError`: it used to hand back an unquantized model carrying a quantized model's provenance
- A per-layer `weight_bits` dict asking for `8` inside a module the same dict leaves at `16` now raises `ValueError`: the FX flow honors the outer `16`, so the `8` could never be applied
- `Quantizer(method='dynamic', weight_bits={...})` now raises `ValueError`. `quantize_dynamic` rewrites every eligible layer off a type list and reads no per-module configuration, so the dict was accepted, ignored, and then recorded as provenance. `method='static'` and `method='qat'` honor one
- `backend='torchao'` now refuses a model whose only `nn.Linear` layers are ones torchao skips (such as a bare `MultiheadAttention`), which used to be quantized into zero layers and tagged as quantized

### Bug Fixes
- `Quantizer(backend='torchao', verbose=True)` reported "quantized 0 layers" on every model: it probed a `weight.layout_type` attribute torchao no longer defines. It now reads the class of the weight

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

