## MODIFIED Requirements

### Requirement: Backend is a builder parameter defaulting to Faer
Every estimator builder SHALL expose a chainable `backend` method accepting
`SKBackendKind` and defaulting to `SKBackendKind::Faer`; `build()` SHALL carry
the configured value into the estimator. The backend SHALL stay a runtime value,
never a type parameter of the estimator. The scalar `SKFloat` is the estimator's
only type parameter, and only when its configuration itself carries scalar
values — an imputation marker, a min/max range, a robust quantile bound; an
estimator whose configuration is scalar-free stays non-generic and takes the
scalar per operation through the `SKFloat`-bounded traits. The fitted model is
generic over the scalar.

#### Scenario: Default backend needs no mention
- **WHEN** an estimator is built without calling `backend`
- **THEN** the estimator carries `SKBackendKind::Faer` and all its operations report `faer`

#### Scenario: Backend override flows into the estimator
- **WHEN** an estimator is built with `.backend(SKBackendKind::MatrixMultiply)`
- **THEN** the built estimator reports that kind in every operation it runs
