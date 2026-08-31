## Purpose

Defines the library's fit and transformation contracts: two traits separated by the supervision axis, typed separation between configured estimator and fitted model (immutable, shareable across threads), compile-time impossibility of predicting without fitting, and transformers with output typed by an associated type — a requirement of static pipeline chaining.

## ADDED Requirements

### Requirement: Two fit traits by supervision
The library SHALL expose an unsupervised fit trait receiving only features, and a supervised fit trait receiving features and targets; estimators SHALL implement exclusively the trait(s) compatible with their nature.

#### Scenario: Classifier requires targets
- **WHEN** a supervised estimator is used
- **THEN** its fit method requires targets in the signature, and calls without targets do not compile

#### Scenario: Clusterer rejects targets at compile time
- **WHEN** code tries to provide targets to the fit of an estimator implementing only the unsupervised trait
- **THEN** compilation fails

### Requirement: Configured estimator is separated from the fitted model
The fit result SHALL be a distinct type — the model — sole bearer of the learned state; fit SHALL operate on a shared reference of the configured estimator, keeping it unchanged and reusable for new fits, including simultaneous ones.

#### Scenario: Same estimator feeds parallel fits
- **WHEN** the same configured estimator is used as source for multiple concurrent fits over distinct partitions
- **THEN** all fits progress without mutual exclusion over the estimator and each produces its own model

#### Scenario: Repeated fit with identical hyperparameters is deterministic at the interface level
- **WHEN** the same configured estimator fits the same deterministic data twice
- **THEN** both resulting models are independent instances of the same model type

### Requirement: Predicting before fitting is unrepresentable
Prediction methods SHALL exist only on the fitted model type; the configured estimator type SHALL NOT expose prediction, making incorrect usage a compile-time error.

#### Scenario: Prediction on an unfitted estimator does not compile
- **WHEN** code tries to call prediction directly on the configured estimator
- **THEN** compilation fails due to the method's absence on that type

### Requirement: Fitted models are shareable across threads
Fitted model types SHALL satisfy safe sending and sharing between threads by construction, without an external mutex for concurrent reading.

#### Scenario: One model serves multiple threads simultaneously
- **WHEN** the same fitted model is shared among threads running concurrent predictions
- **THEN** all predictions complete without additional synchronization required from the consumer

### Requirement: Transformer with typed output
The transformation trait SHALL declare the type produced by the transformation as an associated type, allowing consumers and future pipelines to statically validate compatibility between one stage's output and the next stage's input.

#### Scenario: Compatible chaining validates statically
- **WHEN** a pipeline connects a transformer's declared output to another stage's input through the standard conversion accepted by the data boundary
- **THEN** the chaining compiles without runtime verification of intermediate types

#### Scenario: Declared incompatibility fails early
- **WHEN** a transformer's output type cannot convert into the representation required by the next stage
- **THEN** the mismatch is detectable at compile time by the contract's consumer
