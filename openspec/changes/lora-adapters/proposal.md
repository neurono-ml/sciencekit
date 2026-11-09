# Proposal

## Why

Native import enables continued training, but full fine-tuning is expensive and can
overwrite a good base model. Low-rank adaptation (LoRA) freezes the base parameters and
trains a small low-rank update. The PRD records LoRA creation as a future interoperability
step; this change defines the adapter surface for the library's matrix-parameter models.

## What Changes

- Add `SKLoraAdapter<F>`: rank, alpha, A/B matrices and the scaling `alpha / rank`.
- Attach an adapter to a supported frozen base model (initial coverage: linear and logistic
  models with matrix-valued coefficients; tree/neighbour families are unsupported).
- Fit the adapter on new data while the base stays immutable; predict with
  `base + adapter`; support merge into a standalone native model and unmerge.
- Export/import the adapter through the interop state (metadata + A/B tensors).
- Builder-only construction and validation for rank/alpha, feature/target shapes and
  unsupported base families.
- Explicit non-goals: deep neural networks, GPU training, quantization, distributed
  fine-tuning and full-model fine-tuning.
- Companion tests: adapter shape/scaling, frozen-base invariance, fit/predict equivalence,
  merge/unmerge round-trip, export/import, unsupported families and concurrency.

## Capabilities

### New Capabilities

- `lora-adapters`: low-rank adapter configuration, attach/fit/predict, merge/unmerge,
  persistence and validation.

### Modified Capabilities

None.

## Impact

- **Code:** new adapter module in `sciencekit_linear_model` (or a small
  `sciencekit_adapters` crate) plus interop state integration.
- **APIs:** additive.
- **Dependencies:** existing `ndarray`/`sprs`/`rayon`; no new external crate.
- **Downstream:** enables parameter-efficient continued training and the CLI/retraining
  workflows.
- **Acceptance criteria:** frozen-base invariance, correct scaling, merge/unmerge and
  persistence round-trip.
