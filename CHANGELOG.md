# Release notes

<!-- do not remove -->

## Unreleased

### New Features
- `Quantizer(backend='pt2e', qdq_placement='skip_conv_add')` names **where the Q/DQ pairs sit**, a new axis of the precision grammar. It leaves one edge of every residual block unquantized — the addend produced by a single-user convolution partition — so that result reaches its addition at accumulator precision instead of being rounded to INT8 first. The pass edits the graph torch has just ANNOTATED, so the placement the caller asked for is the one the prepared model, a QAT training run and the exported file all carry; QAT trains through it with `Quantizer(backend='pt2e', method='qat', qdq_placement='skip_conv_add')`. A placement that matches nothing raises rather than being recorded as applied, `qdq_stats` reports the edges as `n_unquantized_conv_add`, and `export_qdq` refuses (keeping no file) a graph that contradicts the placement the model records. The option is **opt-in and changes the arithmetic**: post-training, every pair the graph keeps carries the scale it would have had, but the effect on a trained model's accuracy is unmeasured, and what a runtime does with the resulting graph is that runtime's property. The default, `'per_op'`, is byte-for-byte what this backend has always produced
- `Quantizer(backend='torchao', weight_bits={layer: 8|16}, act_bits=16)` applies a per-layer width over the `nn.Linear` layers torchao rewrites, so part of a Linear-heavy model can be held in floating point. Naming every layer `8` produces the same artifact, byte for byte, as the uniform `method='int8_weight_only'` recipe

### Breaking Changes
- `export_qdq(opset_version=...)` now raises `ValueError` when the graph it produced does not declare the opset that was asked for **and cannot be rewritten to it** (see the enhancement below), instead of returning a file at a different opset. `torch.onnx` exports at the exporter's own opset and down-converts afterwards; when the ONNX version converter cannot rewrite an operator it keeps the original opset and says so only in a log line. The error names both opsets, and nothing is kept on disk — not the graph, and not the external-data file it may reference. A graph that declares no opset at all for the default ONNX domain is refused with its own message rather than that one. The default (`18`) is unaffected
- A per-layer `weight_bits` dict naming a layer the backend does not quantize now raises `ValueError` instead of being silently ignored. On the legacy backends the names it accepts are the module types their default qconfig mapping rewrites (read from torch, so it tracks the installed version) plus any module containing one; `nn.Embedding`, `nn.EmbeddingBag`, `nn.LSTM`, `nn.GRU` and `nn.RNN` are not among them — the default flow leaves them in floating point. On `backend='torchao'` the names it accepts are the `nn.Linear` layers torchao itself selects, which excludes `MultiheadAttention.out_proj`
- A per-layer `weight_bits` dict that would leave *every* layer the backend rewrites in floating point now raises `ValueError`: it used to hand back an unquantized model carrying a quantized model's provenance
- A per-layer `weight_bits` dict asking for `8` inside a module the same dict leaves at `16` now raises `ValueError`: the FX flow honors the outer `16`, so the `8` could never be applied
- `Quantizer(method='dynamic', weight_bits={...})` now raises `ValueError`. `quantize_dynamic` rewrites every eligible layer off a type list and reads no per-module configuration, so the dict was accepted, ignored, and then recorded as provenance. `method='static'` and `method='qat'` honor one
- `backend='torchao'` now refuses a model whose only `nn.Linear` layers are ones torchao skips (such as a bare `MultiheadAttention`), which used to be quantized into zero layers and tagged as quantized

### Enhancements
- `QDQStats` (and `qdq_stats`) gained `n_unquantized_conv_add`: the number of `Add` inputs the file reads from a `Conv` with no Q/DQ pair on the way — directly, or through that convolution's own single-consumer activation, which is the same partition the annotation-time pass matched. It is 0 on every graph the default placement produces, and it is what the `qdq_placement` post-condition reads. `QuantSpec` gained `qdq_placement` and `PrecisionCell` gained `qdq_placements`, so both `as_dict()` views carry one more key
- `export_qdq(opset_version=17)` now produces an opset-17 file instead of raising. The dynamo exporter emits opset 18 and the ONNX version converter cannot bring a QDQ graph back down: opset 18 moved `ReduceMean`'s `axes` from an attribute to an input, and the converter asserts on the attribute a freshly exported graph does not have. `export_qdq` moves the axes back itself — values unchanged, in order — and drops the constant that fed them, so the file a parser limited to opset 17 rejected now parses. The rewrite is not taken on trust: the file that was written is read back by `onnx.checker` at the opset it now declares, and built and run by ONNX Runtime, output-for-output identical to the graph as produced, before it replaces it. A `ReduceMean` whose axes are computed while the graph runs, or any operator that genuinely needs the newer opset, is still refused with a message naming it, and nothing is kept on disk. Checking a lowered graph means running it, so that path needs `onnxruntime`; the default (`18`) is byte-for-byte unaffected and needs nothing new. Expect a logged traceback on the way: `torch.onnx` runs the ONNX version converter before `export_qdq` sees the graph, and the converter reports its own failure. That log is expected and is not the outcome — and on the day the rewrite is refused too, it is the first diagnostic to read

### Bug Fixes
- `Quantizer(backend='torchao', verbose=True)` reported "quantized 0 layers" on every model: it probed a `weight.layout_type` attribute torchao no longer defines. It now reads the class of the weight
- `export_qdq` now writes the optional `kernel_shape` attribute on every `Conv` node, resolving each node's weight back through the `DequantizeLinear` that feeds it. The dynamo exporter omits the attribute, and a QDQ graph hands the weight over as a tensor rather than as an initializer, so an ONNX parser that reads spatial dims off initializers only (TensorRT's among them) had nothing to read and rejected the graph. The edit is spec-legal and semantics-preserving: strip the attributes back off and ONNX Runtime returns byte-identical logits, over an unchanged `qdq_stats`. Only `Conv` is patched; `ConvTranspose` and the pooling operators are untouched

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

