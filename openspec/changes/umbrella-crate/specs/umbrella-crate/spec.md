# Spec Delta

## Purpose

Defines the `sciencekit` umbrella crate: its re-export surface, prelude, feature flags and
documentation entry point.

## ADDED Requirements

### Requirement: Unified re-exports

The umbrella crate SHALL re-export the stable public types, traits and functions of every
sub-crate under documented module paths.

#### Scenario: Common types resolve

- **WHEN** a user imports the umbrella crate
- **THEN** the estimator, model, data, error and execution types resolve without adding
  sub-crate dependencies

#### Scenario: Sub-crate modules remain addressable

- **WHEN** a user needs a specific family
- **THEN** the umbrella exposes it under a named module (for example `linear_model`)

### Requirement: Prelude

The umbrella SHALL provide a `prelude` with the traits and types used by typical code
(builders, fit/predict contracts, data views, errors and the main estimators).

#### Scenario: Prelude compiles a basic program

- **WHEN** a user writes a program with `use sciencekit::prelude::*`
- **THEN** it compiles and can build, fit and predict a model

### Requirement: Feature flags

The umbrella SHALL expose feature flags for optional integrations and SHALL compile with
all features disabled and with each feature individually enabled.

#### Scenario: Minimal build

- **WHEN** the umbrella is built with no optional features
- **THEN** only the core crates are compiled

#### Scenario: Integration feature

- **WHEN** an integration feature is enabled
- **THEN** its re-exports are available and the crate builds

### Requirement: Documentation

The umbrella SHALL provide crate-level documentation pointing to the mdBook and grouping
the API by algorithm family.

#### Scenario: Docs build

- **WHEN** documentation is built
- **THEN** the crate-level page and module pages render without broken links
