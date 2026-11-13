# Spec Delta

## Purpose

Defines the design record of every example in the corpus: the dataset and its origin,
the feature selection, the feature engineering, the preprocessing, the ordered pipeline,
the configuration spectrum, the behavior the reader must observe and the sciencekit
capability the example needs.

## ADDED Requirements

### Requirement: Every example has a design record

Every example of the corpus SHALL have a design record that states, in order, the
dataset and its origin, the feature selection, the feature engineering, the
preprocessing, the ordered pipeline, the configuration spectrum, the expected behavior,
the reported output and the sciencekit capability it depends on.

#### Scenario: The record is complete

- **WHEN** a design record is read
- **THEN** it states every stage of the recipe, and no stage is left unstated — "none"
  with a reason is a valid answer, silence is not

#### Scenario: The origin is visible

- **WHEN** a design record is read
- **THEN** it names the scikit-learn example it adapts and the source file it comes
  from, or states that the example is a sciencekit-native extension

### Requirement: The design goes beyond the original on selection, engineering and preprocessing

A design record SHALL state which features reach the estimator and why, what is derived
from the raw columns, which transformations are applied and where they are fitted, and
the order of the stages.

#### Scenario: Selection is stated

- **WHEN** a design record is read
- **THEN** it names the features that reach the estimator and the reason for the choice

#### Scenario: Engineering is stated

- **WHEN** a design record is read
- **THEN** it names the derived features and how they are computed

#### Scenario: Preprocessing is stated with its fitting point

- **WHEN** a design record names a transformation
- **THEN** it states that the transformation is fitted on the training split only

### Requirement: The configuration spectrum is part of the design

A design record SHALL compare at least two settings of the configuration that matters
for its algorithm, and SHALL state the behavior each setting must produce.

#### Scenario: At least two settings compared

- **WHEN** a design record is read
- **THEN** it compares at least two distinct settings of the algorithm's decisive knob

#### Scenario: The effect is explained

- **WHEN** a design record names a setting
- **THEN** it states what the reader must observe and why it happens

### Requirement: The design names the capability it needs

A design record SHALL state whether the sciencekit capability it needs exists today, is
planned in a named crate, or has no sciencekit path at all.

#### Scenario: The state is stated

- **WHEN** a design record is read
- **THEN** it states the availability of the capability it needs and, when it is
  planned, the crate that must grow first

#### Scenario: A family with no path is an omission

- **WHEN** a design record states that no sciencekit path exists
- **THEN** the example is catalogued as omitted with that reason

### Requirement: The catalog derives from the design records

The catalog SHALL be derived from the design records, and every catalogued example
SHALL have a matching design record.

#### Scenario: Every record has a design

- **WHEN** the catalog is validated against the design records
- **THEN** every record has a matching design record and no design record is missing
  from the catalog

#### Scenario: The availability view is generated

- **WHEN** the availability view is produced
- **THEN** it is generated from the design records and agrees with them entry by entry
