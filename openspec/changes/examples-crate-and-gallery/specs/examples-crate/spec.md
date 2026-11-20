# Spec Delta

## Purpose

Defines the `sciencekit_examples` crate: the runnable example corpus of sciencekit, its
workspace integration, the per-example module layout, the naming rules, the plotting
abstraction and the determinism guarantee each example must honor.

## ADDED Requirements

### Requirement: Examples crate is a workspace member and is never published

The workspace SHALL contain a crate named `sciencekit_examples` under `crates/`, built as
a workspace member, and it SHALL declare itself as not publishable, so that no part of
the example corpus reaches crates.io.

#### Scenario: Default build succeeds offline

- **WHEN** the workspace is built with default features in an environment without network access
- **THEN** the `sciencekit_examples` crate compiles successfully

#### Scenario: No mandatory external dependency

- **WHEN** the manifest of `sciencekit_examples` is inspected
- **THEN** the only non-dev mandatory dependency is `sciencekit`

#### Scenario: The crate is not publishable

- **WHEN** the manifest of `sciencekit_examples` is inspected
- **THEN** it declares that the crate is not published

### Requirement: The example corpus is not built by default

The whole surface of the examples crate SHALL be behind a Cargo feature that is not
enabled by default, so that building or testing the library does not compile the example
corpus.

#### Scenario: Default workspace test does not compile the corpus

- **WHEN** `cargo test --workspace` runs with default features
- **THEN** no example target of the examples crate is compiled

#### Scenario: Enabling the feature compiles the corpus

- **WHEN** the examples feature is enabled
- **THEN** the example targets and their companion tests are compiled and the companion
  tests run

### Requirement: One cargo example target per example, named by example id

The crate SHALL expose exactly one cargo example target per catalogued example, each
named exactly the example id, and each target SHALL contain no logic beyond delegating to
the example's library module.

#### Scenario: Target name equals the example id

- **WHEN** the example targets of the crate are listed
- **THEN** every target name is identical to the `id` of a catalogued example

#### Scenario: Target delegates to the library module

- **WHEN** the source of any example target is read
- **THEN** it only calls the example module's entry point, holding no model, data or
  plotting logic

#### Scenario: Running the target reproduces the example

- **WHEN** an example target is executed
- **THEN** it produces the documented output of the corresponding example

### Requirement: A single command runs any example

The repository SHALL provide one command that runs any example by its id from the
repository root, and the command SHALL be the one quoted by the gallery pages.

#### Scenario: The documented command runs the example

- **WHEN** the documented run command is issued with an example id
- **THEN** that example runs and prints its documented output

### Requirement: The example target manifest block is derived

The example target declarations of the crate SHALL live in a marked block of the manifest
that is regenerated from the examples directory, and a test SHALL assert that the
committed block matches the directory.

#### Scenario: The block matches the directory

- **WHEN** the crate's tests run
- **THEN** the declared example targets are exactly the directories present under the
  examples directory

#### Scenario: A new example needs no hand-written manifest line

- **WHEN** an example directory is added
- **THEN** regenerating the block is enough to declare its target

### Requirement: Per-example folder modules with pure dispatchers

Each example family and each example SHALL live in its own folder module, whose `mod.rs`
SHALL declare submodules and re-export public items only, without implementation logic,
and no source file of the crate SHALL exceed 200 lines.

#### Scenario: Dispatcher carries no implementation

- **WHEN** the `mod.rs` of any example family or example is read
- **THEN** it contains only submodule declarations, re-exports and the companion test
  module declaration

#### Scenario: File size limit

- **WHEN** every source file of the crate is measured
- **THEN** no file exceeds 200 lines

### Requirement: Example identifiers avoid abbreviations

Every example id, module name, public item and variable of the crate SHALL use full
English words in snake_case, without abbreviations; canonical mathematical symbols that
are the standard notation of the algorithm MAY appear as tokens.

#### Scenario: No abbreviated identifiers

- **WHEN** the identifiers of the crate are inspected
- **THEN** no identifier abbreviates a word, except the canonical mathematical tokens
  listed in the porting guide

#### Scenario: Identifier names agree

- **WHEN** an example id is compared with the name of its module and its example target
- **THEN** all three are the same identifier

### Requirement: Deterministic example output

Every example SHALL be deterministic given the seed it documents, and SHALL NOT depend on
ambient state such as thread scheduling, wall-clock time or unseeded randomness.

#### Scenario: Repeated runs agree

- **WHEN** the same example target is executed twice
- **THEN** both runs report the same metrics and the same numbers

#### Scenario: Seed is explicit

- **WHEN** an example builds a stochastic component
- **THEN** the seed is passed explicitly by the example, not inherited from a global
  default

### Requirement: Plotting is behind an internal abstraction

Every figure produced by an example SHALL be produced through the crate's internal
plotting abstraction, no example module other than the plotting module SHALL depend on a
plotting library, and the `scienceview`-backed implementation SHALL live behind a
non-default Cargo feature.

#### Scenario: Crate builds without the plotting backend

- **WHEN** the crate is built with default features
- **THEN** it compiles and its tests pass without any plotting backend enabled

#### Scenario: Plotting code is isolated

- **WHEN** the dependencies of the example modules are inspected
- **THEN** only the plotting module depends on a plotting library

#### Scenario: Backend is opt-in

- **WHEN** the manifest of the crate is inspected
- **THEN** the feature enabling the `scienceview` backend is not enabled by default

### Requirement: Every example ships its companion tests

Each example module SHALL declare a companion `*_tests.rs` module containing the test
that exercises the example's own pipeline, and no global `tests/` directory SHALL exist in
the crate.

#### Scenario: Companion test module exists

- **WHEN** an example module is inspected
- **THEN** a companion `*_tests.rs` module beside it exercises the example's pipeline

#### Scenario: No global test directory

- **WHEN** the crate's directory tree is inspected
- **THEN** no top-level `tests/` directory exists
