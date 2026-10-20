## MODIFIED Requirements

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
