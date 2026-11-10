# Spec Delta

## Purpose

Defines the dataset surface owned by the examples crate: the synthetic generators and the
small classic datasets the examples rely on, together with the determinism and
no-network guarantees they must honor.

## ADDED Requirements

### Requirement: Datasets are owned by the examples crate

The dataset surface SHALL be provided by the examples crate itself, as a module of that
crate, and SHALL NOT depend on an external datasets crate or on any other sciencekit
crate that does not already exist.

#### Scenario: Dataset module lives in the crate

- **WHEN** the dataset surface is located
- **THEN** it is a module of `sciencekit_examples`

#### Scenario: No dependency on a future datasets crate

- **WHEN** the manifest of the examples crate is inspected
- **THEN** it declares no dependency on a dataset-providing sciencekit crate

### Requirement: Synthetic generators mirror the scikit-learn generators

The dataset module SHALL provide synthetic generators equivalent to the scikit-learn
`make_*` generators the gallery depends on, covering classification, blobs, circles,
moons, regression, swiss roll, sparse and multilabel designs, and SHALL be able to
produce both dense and sparse designs where scikit-learn does.

#### Scenario: Generator produces the documented shape

- **WHEN** a generator is called with a documented sample count, feature count and seed
- **THEN** it returns a design with the requested shape and a target with the requested
  cardinality

#### Scenario: Sparse design is produced without densifying

- **WHEN** a generator is asked for a sparse design
- **THEN** the returned design is a sparse view and is never materialized as dense

### Requirement: Classic datasets are embedded

The dataset module SHALL embed the small classic datasets the gallery relies on — iris,
digits, wine, diabetes, breast cancer and linnerud — with their real shapes, feature
names and target names.

#### Scenario: Embedded dataset has the expected shape

- **WHEN** an embedded classic dataset is loaded
- **THEN** it returns the documented number of samples, features, classes and targets

#### Scenario: Feature and target names are preserved

- **WHEN** an embedded classic dataset is loaded
- **THEN** the feature names and the target names of the dataset are available

### Requirement: Datasets never touch the network

Loading or generating a dataset SHALL NOT perform any network access, and every dataset
SHALL be obtainable offline.

#### Scenario: Offline load

- **WHEN** a dataset is loaded in an environment without network access
- **THEN** the load succeeds

### Requirement: Datasets are deterministic and seedable

Every generator SHALL accept an explicit seed and produce the same design for the same
seed, and every embedded dataset SHALL be byte-identical across runs.

#### Scenario: Same seed, same design

- **WHEN** a generator is called twice with the same seed and parameters
- **THEN** both results are equal element-wise

#### Scenario: Different seed, different design

- **WHEN** a generator is called with two different seeds
- **THEN** the resulting designs differ

### Requirement: Every embedded dataset ships its provenance

Each embedded dataset SHALL ship a provenance record naming its original source, its
license, its citation and the transformations applied when it was embedded, and the
dataset module SHALL assert that every embedded dataset has one.

#### Scenario: Provenance exists for every dataset

- **WHEN** the dataset module is validated
- **THEN** every embedded dataset has a provenance record naming its source and license

#### Scenario: Transformations are recorded

- **WHEN** the provenance record of an embedded dataset is read
- **THEN** it states the transformations applied when the data was embedded

### Requirement: Datasets may carry missing values

The dataset module SHALL be able to produce designs with missing values so that the
imputation examples have a source of missingness, and SHALL document how the missingness
was introduced.

#### Scenario: Missing values are introduced on request

- **WHEN** a generator is asked to introduce a missingness fraction
- **THEN** the returned design contains that fraction of missing entries, and the
  documented mechanism is recorded for the example
