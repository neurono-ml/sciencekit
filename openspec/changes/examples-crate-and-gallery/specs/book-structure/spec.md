# Spec Delta

## Purpose

Defines the didactic path of the sciencekit book: the chapter taxonomy, the contract
for the complete example every algorithm chapter carries, and the index that makes
every example in the corpus findable.

## ADDED Requirements

### Requirement: The book follows one didactic path

The book SHALL be organized as a single path that starts from what sciencekit is and
why it exists, continues through one complete example from raw arrays to a saved model
and a command-line inference, presents the algorithm families, and closes with the
choice of an estimator, interoperability and the index of examples.

#### Scenario: The chapters follow the path

- **WHEN** the book's summary is read
- **THEN** the chapters appear in the didactic order: what sciencekit is, the complete
  example, the algorithms, choosing an estimator, interoperability, the examples index

#### Scenario: Every chapter belongs to a part

- **WHEN** a chapter is listed in the summary
- **THEN** it belongs to exactly one part of the path

### Requirement: The algorithms part covers every family with its own subsection

The algorithms part SHALL have a subsection for preprocessing and pipelines, for
supervised learning, for unsupervised learning, for training observability, for lazy
loading, for export, for inference, and for the choice of an estimator, and each
algorithm SHALL have its own level-four subsection inside its family.

#### Scenario: Each algorithm has its own subsection

- **WHEN** a family of algorithms is presented
- **THEN** every algorithm of the family has a level-four subsection of its own

#### Scenario: The families are separated by learning paradigm

- **WHEN** the supervised and unsupervised families are read
- **THEN** the supervised family separates classification from regression, and the
  unsupervised family separates clustering from dimensionality reduction

### Requirement: Every algorithm chapter carries a complete example

An algorithm subsection SHALL show the algorithm *in use*: its purpose, the data it
expects, a complete example with feature engineering, feature selection, preprocessing
and a pipeline where those apply, the configuration spectrum worth comparing, and the
behavior the reader must observe.

#### Scenario: The example is complete

- **WHEN** an algorithm subsection is read
- **THEN** it shows a runnable example and, where the algorithm admits them, the
  feature selection, the feature engineering, the preprocessing and the pipeline

#### Scenario: The configuration is not a single value

- **WHEN** an algorithm subsection is read
- **THEN** it compares at least two configurations of the algorithm and states what
  each one changes

### Requirement: The complete example chapter is end to end

The book SHALL carry one worked example that runs from in-memory arrays through a
real dataset, a lazy dataset, a training run with exported metrics, a configuration
comparison, a pipeline, an exported artifact and a command-line inference.

#### Scenario: Every stage of the example is present

- **WHEN** the complete example chapter is read
- **THEN** it contains the stages in order: arrays, a real dataset, a lazy dataset,
  exported training metrics, the configuration comparison, the pipeline, the exported
  artifact and the command-line inference

#### Scenario: The split discipline is stated

- **WHEN** the complete example chapter is read
- **THEN** it states how the data is separated into training and test, and which
  validation techniques it uses

### Requirement: The examples index lists every example used

The book SHALL close with an index of every example the book uses, each linking to its
page in the gallery and to the chapter that teaches it.

#### Scenario: Every example used is listed

- **WHEN** the examples index is compared with the examples cited by the chapters
- **THEN** every cited example appears in the index and no index entry is uncited

#### Scenario: Each entry is findable

- **WHEN** an index entry is read
- **THEN** it links to the example's gallery page and to the chapter that uses it
