# Spec Delta

## Purpose

Defines the export/import coverage of every fitted model family: which tensors and metadata
each model exposes, how stable names are formed and when native import is supported.

## ADDED Requirements

### Requirement: Every fitted model exports state

Every fitted model and pipeline model SHALL implement `SKExportable<F>` and return a state
whose tensor names and metadata are sufficient to describe the learned model.

#### Scenario: Preprocessing models export

- **WHEN** a fitted scaler, encoder, imputer or indicator is exported
- **THEN** its statistics/tables are present as named tensors and its hyperparameters are
  present as metadata

#### Scenario: Supervised models export

- **WHEN** a fitted linear model, neighbour, SVM, tree, forest or booster is exported
- **THEN** its coefficients/support vectors/tree arrays and class table are present

#### Scenario: Unsupervised models export

- **WHEN** a fitted clusterer, decomposition or outlier detector is exported
- **THEN** its centers/components/tree state and resolved configuration are present

### Requirement: Stable tensor names

Each model family SHALL document a stable naming scheme, and the same fitted model SHALL
export the same names across runs and library versions within the documented compatibility
window.

#### Scenario: Repeated export uses the same names

- **WHEN** the same fitted model is exported twice
- **THEN** the tensor-name sets are identical

#### Scenario: Family naming is documented

- **WHEN** a family is retrofitted
- **THEN** its proposal/design records the names and their meaning

### Requirement: Metadata completeness

The state SHALL include the estimator name, format version, scalar dtype, feature count,
class/label tables when applicable, the resolved backend/solver and the hyperparameters
needed to reconstruct the configuration.

#### Scenario: Backend is preserved

- **WHEN** a model fitted with a non-default backend is exported
- **THEN** the metadata records that backend

#### Scenario: Class table is preserved

- **WHEN** a classifier is exported
- **THEN** the metadata contains its canonical class table

### Requirement: Native import where reconstructible

Models whose state is sufficient SHALL implement `SKImportable<F>` and reconstruct an
equivalent model; models that cannot SHALL return `UnsupportedOperation` with guidance.

#### Scenario: Round-trip prediction

- **WHEN** a supported model is exported, imported and used to predict
- **THEN** predictions match the original within the documented tolerance

#### Scenario: Unsupported model is explicit

- **WHEN** a model cannot be reconstructed from its state
- **THEN** import returns `UnsupportedOperation`, never a partial model

### Requirement: Pipeline and grid-search export

A fitted pipeline SHALL export its fitted stages and final model; a grid-search model SHALL
export the best fitted model and its selection metadata.

#### Scenario: Pipeline export nests stages

- **WHEN** a fitted pipeline is exported
- **THEN** each stage is present under a stage-qualified name and the final model is
  present

#### Scenario: Grid-search export

- **WHEN** a grid-search model is exported
- **THEN** the best candidate index, score and refitted model are present

### Requirement: Concurrency and validation

Export SHALL be read-only and deterministic under concurrent calls; import SHALL validate
names, versions, shapes and dtypes before constructing a model.

#### Scenario: Concurrent export agrees

- **WHEN** several threads export the same model
- **THEN** every state equals the single-threaded state

#### Scenario: Corrupt tensor is rejected

- **WHEN** an imported tensor has the wrong shape
- **THEN** import returns `ShapeMismatch` and no model is produced
