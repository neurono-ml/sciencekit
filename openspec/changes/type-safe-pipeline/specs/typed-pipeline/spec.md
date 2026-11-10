# Spec Delta

## Purpose

Defines the linear typed pipeline: stage composition, compile-time scalar/step compatibility,
fitting, prediction, representation flow and validation.

## ADDED Requirements

### Requirement: Builder-only typed composition

`SKPipeline` SHALL be constructed through a chain builder whose type records the composed
stages; each `then` SHALL accept a transformer stage and the finalizer SHALL accept a
supervised estimator.

#### Scenario: Two transformers and a final estimator

- **WHEN** a scaler, an imputer and a classifier are composed
- **THEN** the resulting type encodes all three stages and the classifier prediction
  surface

#### Scenario: Empty pipeline is rejected

- **WHEN** a pipeline is finalized without any transformer
- **THEN** construction returns `InvalidHyperparameter`

### Requirement: Compile-time compatibility

The builder SHALL reject at compile time a stage whose scalar differs from the pipeline
scalar and a final estimator whose input cannot consume the last stage's output.

#### Scenario: Mixed scalars do not compile

- **WHEN** an `f32` pipeline is composed with an `f64` stage
- **THEN** compilation fails

#### Scenario: An estimator of the wrong supervision kind does not compile

- **WHEN** a final estimator cannot consume the last transformer output
- **THEN** compilation fails

### Requirement: Fit

Fit SHALL fit every transformer on the current training data, transform it forward, and
fit the final estimator on the last block; the fitted model SHALL store every fitted
transformer model and the final model.

#### Scenario: Hand-checkable chain

- **WHEN** a scaler followed by an imputer and a classifier is fitted
- **THEN** each stage is fitted on the previous stage's output and the classifier sees the
  imputed, scaled features

#### Scenario: Repeated fit

- **WHEN** the same pipeline is fitted twice on the same data
- **THEN** the fitted stages and final predictions are identical

### Requirement: Prediction

The fitted classifier pipeline SHALL implement `SKClassifierPredictor` and delegate
`predict_labels`/`predict_probabilities` to the final model after transforming the input;
the regressor pipeline SHALL implement `SKRegressorPredictor`.

#### Scenario: Classifier labels and probabilities

- **WHEN** a fitted classifier pipeline predicts
- **THEN** its labels and probabilities equal transforming the input and calling the final
  model

#### Scenario: Regressor responses

- **WHEN** a fitted regressor pipeline predicts
- **THEN** its responses equal transforming the input and calling the final model

### Requirement: Representation flow

Dense and sparse blocks SHALL flow through the chain without losing their representation;
a stage may change dense to sparse or sparse to dense according to its own contract, and no
stage SHALL densify a sparse input it declares to accept.

#### Scenario: Sparse chain

- **WHEN** a sparse input passes through sparse-capable stages
- **THEN** the block stays CSR and no dense copy is made

#### Scenario: Representation change is explicit

- **WHEN** a stage outputs a dense block after a sparse input
- **THEN** the change is the stage's documented output, not an implicit pipeline
  densification

### Requirement: Model accessors

The fitted pipeline SHALL expose the fitted stages, the final model, the stage count and
the final feature count.

#### Scenario: Accessors describe the chain

- **WHEN** a pipeline is fitted
- **THEN** the fitted stage list has the configured length and the final model is
  accessible

### Requirement: Validation and concurrency

The pipeline SHALL reject runtime feature-count mismatches reported by stages and SHALL be
safely shareable, producing the same predictions under concurrent calls.

#### Scenario: Stage mismatch is surfaced

- **WHEN** a stage rejects the incoming feature count
- **THEN** the pipeline returns that structured error

#### Scenario: Concurrent prediction agrees

- **WHEN** several threads predict with the same fitted pipeline
- **THEN** every result equals the sequential result
