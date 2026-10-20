# backend-selection Specification

## Purpose
Declares the single, uniform way estimators expose the math backend: as an
explicit builder parameter of the closed `SKBackendKind` enum, defaulting to
`Faer`. The backend travels from builder to estimator to fitted model, and
every public operation records it in its observability span. Uniformity is
deliberate: even estimators whose math never touches heavy algebra (scalers
run only reductions) carry the parameter, so all algorithms share one
selection surface and future backends (BLAS, GPU) plug in once, centrally.

## Requirements

### Requirement: Backend is a builder parameter defaulting to Faer
Every estimator builder SHALL expose a chainable `backend` method accepting
`SKBackendKind` and defaulting to `SKBackendKind::Faer`; `build()` SHALL carry
the configured value into the estimator. The backend SHALL stay a runtime value,
never a type parameter of the estimator.

#### Scenario: Default backend needs no mention
- **WHEN** an estimator is built without calling `backend`
- **THEN** the estimator carries `SKBackendKind::Faer` and all its operations report `faer`

#### Scenario: Backend override flows into the estimator
- **WHEN** an estimator is built with `.backend(SKBackendKind::MatrixMultiply)`
- **THEN** the built estimator reports that kind in every operation it runs

#### Scenario: The scalar is the estimator's only possible type parameter
- **WHEN** an estimator is built
- **THEN** the backend never appears as a type parameter: the scalar `SKFloat` is the only possible one, and only for a configuration that carries scalar values, while a scalar-free configuration stays non-generic and takes the scalar per operation

### Requirement: Every public operation records the configured backend
Every `fit`, `transform`, `predict`, `partial_fit` and `score` operation SHALL
set `SKOperationAttributes.backend` to the configured kind — never a hardcoded
value — so traces always reflect the consumer's selection.

#### Scenario: Span reflects the override
- **WHEN** `fit` runs on an estimator configured with a non-default backend
- **THEN** the emitted `fit` span carries that backend kind

### Requirement: Fitted models carry the backend
The `fit` operation SHALL transfer the estimator's backend kind into the
fitted model, and every model-side operation (`transform`, `predict`, `score`)
SHALL record the carried kind in its span. A model SHALL NOT re-resolve or
invent a backend.

#### Scenario: Model inherits the estimator backend
- **WHEN** a model is produced by `fit` on an estimator with a non-default backend
- **THEN** the model's `transform` span carries that same backend kind

### Requirement: Dispatch on the kind at operation time
Algorithm crates SHALL resolve the configured kind to a concrete
`SKMathBackend` at operation time (e.g. via `match` selecting `SKFaerBackend`,
`SKMatrixMultiplyBackend`, or the opt-in `blas-backend` build), inside the
operation that needs heavy algebra. Resolution SHALL NOT happen in `build()`:
like the execution plan, the backend binding is per-operation.

#### Scenario: Heavy algebra routes through the selected backend
- **WHEN** an operation requiring GEMM or a decomposition runs with a configured backend
- **THEN** the computation executes on the concrete backend matching the kind

### Requirement: Closed enum, no custom-backend injection
`SKBackendKind` SHALL remain the only selection surface: estimators SHALL NOT
accept generic `B: SKMathBackend<F>` parameters nor boxed trait objects.
Third-party custom backends are out of scope; adding a backend means extending
the enum, its dispatch, and this spec together.

#### Scenario: New backend arrives centrally
- **WHEN** a new backend (BLAS, GPU) is introduced
- **THEN** it appears as a new `SKBackendKind` variant with dispatch coverage and an updated spec, and every estimator exposes it with no signature change
