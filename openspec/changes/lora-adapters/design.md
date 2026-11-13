# Design

## Context

See `proposal.md` — Why. The library has matrix-parameter linear/logistic models; no neural
networks. LoRA is therefore scoped to matrices such as coefficient blocks, not arbitrary
deep layers. The adapter reuses ndarray/sprs and the existing optimizer substrate.

## Goals / Non-Goals

**Goals:**
- A small adapter with frozen base parameters.
- Correct merge/unmerge and persistence.
- A clear initial coverage (linear/logistic matrix parameters).

**Non-Goals:**
- Deep neural networks, GPU training, quantization, distributed fine-tuning.
- Full-model fine-tuning.
- Adapters for tree/neighbour models.

## Decisions

### Decision 1 — Low-rank delta on selected matrices

For a base parameter matrix `W` (`p × d`), the adapter stores `A` (`r × d`) and `B`
(`p × r`), and applies `W' = W + (alpha / r) · B A`. Only matrices selected by the target
configuration are adapted. The base model is borrowed immutably.

### Decision 2 — Fitting

The adapter fits only `A`/`B`. Initial coverage:
- linear regression: alternating least squares / gradient descent on the residual;
- logistic regression: gradient descent on the adapter parameters using the existing
  logistic objective.

The optimizer, tolerance and iteration budget are builder parameters. Other base families
return `UnsupportedOperation`.

### Decision 3 — Prediction and merge

Prediction computes the base output plus the adapter contribution. `merge` produces a
standalone native model with `W'`; `unmerge` returns the original `W` and the adapter.
Merge/unmerge tests assert prediction parity.

### Decision 4 — Persistence

Adapter state uses the interop contract: metadata (`rank`, `alpha`, target parameter) and
`A`/`B` tensors under `lora.<target>.a`/`.b`. Import reconstructs the adapter and validates
shapes.

### Decision 5 — File layout

```
crates/sciencekit_linear_model/src/adapters/lora/
  mod.rs, builder.rs, config.rs, core_implementation.rs, fit.rs, predict.rs,
  merge.rs, state.rs, *_tests.rs
```

`mod.rs` is a pure dispatcher; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Non-convex adapter fit** → deterministic initialization, iteration budget and
  convergence tests; document that it is a local optimum.
- **Limited applicability** → initial linear/logistic coverage; explicit unsupported
  families.
- **Merge numerical differences** → tolerance-based parity tests.
- **Naming confusion with neural LoRA** → docs state the scope explicitly.

## Migration Plan

Additive: new adapter module. No existing API changes.

## Open Questions

None blocking. Deep-model LoRA is a future change if neural estimators arrive.
