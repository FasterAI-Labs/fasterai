# Dimensional `Quantizer` — design north-star

**Status:** design spec (target architecture). AdaRound is the first concrete piece built toward it.

## Principle

fasterai's pruning API is a small set of **composable, orthogonal dimensions** — `Sparsifier(granularity, context, criteria, schedule)`. Every pruning technique is a *value on an axis*, not a bespoke function. This composability (one grammar, sensitivity-driven) is the moat, more than any single technique.

Quantization has the **same dimensional structure**. This spec defines a `Quantizer` that mirrors `Sparsifier`, so a user who knows one knows the other, and every quant technique — existing (`quantize_mixed`, SmoothQuant) and new (AdaRound, GPTQ) — is a composable value, not a separate feature.

## The dimensions (4 shared with pruning + 1 new)

| dimension | Pruning (`Sparsifier`) | Quantization (`Quantizer`) |
|---|---|---|
| **granularity** — scope a scale is shared over | weight / vector / kernel / filter | `tensor` / `channel` / `group` / `block` |
| **context** — allocation across layers | local / global | `uniform` / `mixed` (per-layer bits, sensitivity-driven) |
| **criteria** — how values are decided | large_final / movement / gradient / Wanda | `minmax` / `percentile` / `mse` / `adaround` / `gptq` |
| **schedule** — when it is applied | one_shot / iterative / agp / cos | `None` (PTQ) / a `Schedule` (progressive QAT) |
| **precision** — how aggressive *(new; no pruning analog)* | — | `int8` / `int4` / `int16` / `fp16` / `bf16`, or `{layer: dtype}` for mixed |

## Target API

```python
class Quantizer:
    "Reduce numerical precision along composable dimensions — the quantization analogue of Sparsifier."
    def __init__(self,
        model: nn.Module,
        granularity: str = 'channel',            # 'tensor'|'channel'|'group'|'block'
        context:     str = 'uniform',            # 'uniform' | 'mixed' (per-layer, sensitivity-driven)
        criteria:    QuantCriteria = minmax,     # minmax|percentile|mse|adaround|gptq
        schedule:    Schedule | None = None,     # None -> PTQ one-shot; Schedule -> progressive QAT
        dtype: str | dict[str, str] = 'int8',    # 'int8'|'int4'|'int16'|'fp16'|'bf16'  OR  {layer: dtype} for mixed
        *,
        act_dtype: str | None = 'int8',          # activation precision (None -> weight-only, e.g. W4A16)
        symmetric: bool = True,                  # symmetric vs asymmetric zero-point
        dynamic:   bool = False,                 # activations: dynamic (per-inference) vs static (calibrated)
        group_size: int = 128,                   # block width for 'group'/'block' granularity (INT4)
        layer_type: type = nn.Conv2d,            # which modules to quantize (like Sparsifier)
    ): ...

    def quantize_model(self, calibration_dl=None) -> nn.Module:
        "PTQ: calibrate then apply. For QAT, use QuantizeCallback(schedule=...)."
```

`dtype` is **a value or a per-layer dict** — exactly like `Sparsifier.sparsity`. That single choice makes uniform vs mixed-precision fall out of one API.

## `QuantCriteria` — the calibration strategy (mirrors pruning `Criteria`)

```python
class QuantCriteria:
    "How to choose (scale, zero_point) for a tensor — the quant analogue of a pruning Criteria."
    needs_data: bool                                  # needs calibration_dl? (adaround/gptq/percentile -> yes)
    def params(self, w, n_bits, granularity, symmetric, calib=None) -> tuple[Tensor, Tensor]: ...

minmax     = QuantCriteria(...)   # min/max range — closed-form, data-free for weights
percentile = lambda p=99.9: ...   # clip outliers at the p-th percentile
mse        = QuantCriteria(...)   # scale minimising quantization MSE
adaround   = QuantCriteria(...)   # learned per-weight rounding (needs calib) — the "make INT4 safe" criterion
gptq       = QuantCriteria(...)   # OBC / Hessian-guided reconstruction (needs calib)
```

## Every technique is a config, not a feature

| technique | how it is expressed |
|---|---|
| per-channel INT8 (near-lossless default) | `granularity='channel', dtype='int8'` |
| per-tensor (naive baseline) | `granularity='tensor'` |
| per-group INT4 weight-only | `granularity='group', dtype='int4', act_dtype=None, group_size=128` |
| **AdaRound** | `criteria=adaround` |
| **GPTQ** | `criteria=gptq` |
| percentile / MSE calibration | `criteria=percentile()` / `criteria=mse` |
| **mixed-precision** (was `quantize_mixed()`) | `context='mixed'` -> per-layer `dtype` dict from sensitivity |
| W4A16 | `dtype='int4', act_dtype='fp16'` |
| dynamic quant | `dynamic=True` |
| data-free | pass a *synthetic* `calibration_dl` (BN-stat-generated) — orthogonal to the axes |
| **SmoothQuant** | a pre-quant **transform** (like `BN_Folder`), not a Quantizer dim |
| ~~CLE / Bias-Correction~~ | pre-quant transform — **validated redundant with per-channel; not integrated** |

## Mechanism / decision split (same as pruning)

- **fasterai `Quantizer` = mechanism.** Applies quantization at a given granularity/dtype/criteria; accepts `dtype` as a **per-layer dict** when `context='mixed'`. It does not decide the allocation.
- **fasterrecipes = decision.** For `context='mixed'`, it calls sensitivity analysis to produce the per-layer bit-width dict, then hands it to the Quantizer — exactly how `Sparsifier` takes a sparsity dict that fasterrecipes computes.

**The same sensitivity engine feeds both pruning `layer_targets` and quantization bit-width** -> one allocation brain, two techniques -> cross-technique co-optimization (allocate ratio *and* bits per layer from one sensitivity pass). This is the differentiator neither Pruna nor Embedl has.

## Validated evidence behind the design

- **per-channel INT8 is already near-lossless** (−0.4 pt on MobileNetV2). "Safe INT8" is solved by per-channel granularity; it needs no help.
- **AdaRound makes INT4 viable — but architecture-dependently.** ResNet-class: INT4+AdaRound ≈ INT8 (~1 pt). MobileNet-class: still ~57 pt below INT8 (depthwise convs are the INT4-killer). => INT4 is the "up to ~8× smaller at ~1 pt" lever *for redundant CNNs*, and `context='mixed'` (route depthwise to INT8) is how efficient nets are handled.
- **INT4 is a size lever, not a CPU-speed one** (our own benchmark: W4A32 is ~4× *slower* on CPU, no fast kernel). Market INT4 as size/memory.
- **CLE / Bias-Correction: validated redundant** with per-channel (within noise) and a −24 pt ReLU6 landmine — *not* integrated.

## Migration path (incremental, non-breaking)

The current `Quantizer` is backend-oriented (`backend`, `method`). We do **not** big-bang refactor it. Instead:
1. Add `AdaRound` as a self-contained criterion + a thin apply path (this branch) — shaped to become `criteria='adaround'`.
2. Revive mixed-precision (`quantize_mixed`) on current master as `context='mixed'`, fed by sensitivity.
3. Add SmoothQuant as a pre-quant transform.
4. Unify under the dimensional `Quantizer` once the pieces exist.

## Honest constraint (encoded, not hidden)

The API can *express* combinations that don't deploy fast (per-group INT4 on CPU, mixed-precision on a runtime that ignores it). That is intentional: `quantize_model` gives faithful **accuracy** (fake-quant), and `benchmark_on_device` (fasterrecipes) says which combo actually **pays off** on the target. Express anything; verify what ships.
