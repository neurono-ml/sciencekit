# Spec Delta

## MODIFIED Requirements

### Requirement: Every public operation records the configured backend
Every `fit`, `transform`, `predict`, `partial_fit` and `score` operation SHALL
set `SKOperationAttributes.backend` to the configured kind — never a hardcoded
value — so traces always reflect the consumer's selection. This obligation binds
estimator and fitted-model operations. Standalone metric and scoring functions are pure
computations with no configured backend and SHALL NOT resolve or record one.

#### Scenario: Span reflects the override
- **WHEN** `fit` runs on an estimator configured with a non-default backend
- **THEN** the emitted `fit` span carries that backend kind

#### Scenario: Standalone metrics are backend-free
- **WHEN** a metric function evaluates stored predictions and targets outside any estimator
- **THEN** no backend is resolved or recorded
