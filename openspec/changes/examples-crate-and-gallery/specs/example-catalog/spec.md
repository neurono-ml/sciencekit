# Spec Delta

## Purpose

Defines the machine-readable catalog that tracks parity with the scikit-learn example
gallery: one record per gallery entry, its scikit-learn source, its dataset, the
algorithms and configuration it exercises, the languages it covers and its porting status.

## ADDED Requirements

### Requirement: Catalog covers every scikit-learn gallery entry

The catalog SHALL contain exactly one record for each entry of the scikit-learn example
gallery, including the entries that are intentionally not ported.

#### Scenario: Every gallery entry is accounted for

- **WHEN** the catalog is compared against the gallery index
- **THEN** each gallery entry has exactly one record

#### Scenario: Omissions are recorded, not dropped

- **WHEN** an entry is not ported
- **THEN** its record states the reason and the status is an explicit omission

### Requirement: Record content is fixed and machine-readable

Each record SHALL carry the example id, the family, the scikit-learn source file and its
documentation URL, the dataset source, the sciencekit crates, the estimators, the
configuration knobs, the supported languages and the status, and the catalog SHALL be
stored in a machine-readable format validated by the build.

#### Scenario: Required fields are present

- **WHEN** a record is validated
- **THEN** every required field is present and non-empty

#### Scenario: Validation runs in the build

- **WHEN** the crate's tests run
- **THEN** the catalog is parsed and every record is validated

### Requirement: Every ported example traces to its scikit-learn source

A ported example SHALL name the exact scikit-learn example file it adapts, and SHALL
record the dataset source and the estimators used by that source file.

#### Scenario: Source path is recorded

- **WHEN** a ported record is read
- **THEN** it names the scikit-learn gallery file it is ported from

#### Scenario: Dataset source is recorded

- **WHEN** a ported record is read
- **THEN** it names the scikit-learn dataset or generator the source file loads

### Requirement: Configuration spectrum is recorded

A ported example SHALL record the configuration knobs it exercises and the number of
distinct settings compared, so that the didactic contract of the example is verifiable
rather than asserted.

#### Scenario: Knobs are enumerated

- **WHEN** a ported record is read
- **THEN** it lists the configuration parameters varied by the example

#### Scenario: At least one comparison

- **WHEN** a ported record declares its knobs
- **THEN** it compares at least two distinct settings of the algorithm

### Requirement: Status values are meaningful

The status of a record SHALL distinguish ported, partially ported, blocked by a missing
sciencekit capability, and intentionally omitted, and a blocked record SHALL name the
missing capability.

#### Scenario: Blocked record names the blocker

- **WHEN** a record is blocked by a missing sciencekit capability
- **THEN** it names that capability, so the corpus growth follows the algorithm roadmap

### Requirement: Catalog drives the gallery index

The gallery index of the "Examples" documentation section SHALL be generated from the
catalog, so that a record and its page cannot drift apart.

#### Scenario: Index is generated

- **WHEN** the gallery index is built
- **THEN** its entries are derived from the catalog records

#### Scenario: Catalog and index agree

- **WHEN** the catalog is validated against the built index
- **THEN** every ported record has one page and every page has one record
