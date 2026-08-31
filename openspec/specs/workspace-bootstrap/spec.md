# workspace-bootstrap Specification

## Purpose
Establishes the verifiable foundation of the sciencekit repository: a pinned, reproducible Rust toolchain; an automated quality gate in CI for every pull request; and the declared Apache-2.0 license. Every future change assumes these behaviors as preconditions.
## Requirements
### Requirement: Pinned, reproducible toolchain
The repository SHALL pin the exact Rust toolchain version (1.85) declaratively, so that any clean clone compiles with the same version, using edition 2024, without depending on the contributor's installed Rust.

#### Scenario: Clean clone uses the correct toolchain
- **WHEN** a clean clone of the repository runs any Cargo command without additional configuration
- **THEN** the exact 1.85 toolchain is selected automatically from the declaration versioned in the repository

#### Scenario: Code incompatible with edition 2024 is rejected at build
- **WHEN** source code requiring an edition earlier than 2024 is compiled in the project environment
- **THEN** compilation fails, evidencing that edition 2024 is in force

### Requirement: CI quality gate on every pull request
CI SHALL automatically run, on each pull request, formatting checks, static analysis with warnings treated as errors, the complete workspace test suite (including documentation tests) and example build/tests. Any failure SHALL prevent automatic approval of the change.

#### Scenario: Divergent formatting fails the PR
- **WHEN** a pull request contains code outside the defined formatting standard
- **THEN** the formatting check job fails and points out the divergences

#### Scenario: Static analysis warning fails the PR
- **WHEN** the code introduces any static analysis warning
- **THEN** the analysis job fails, because warnings are promoted to errors

#### Scenario: Test failure fails the PR
- **WHEN** any workspace or example test fails on CI
- **THEN** the pull request cannot be considered validated by automation

### Requirement: Guaranteed MSRV compatibility
CI SHALL include a dedicated check compiling and testing the project using exclusively the minimum supported version (1.85), so that accidental use of newer Rust features is detected before merge.

#### Scenario: Feature newer than the MSRV is detected
- **WHEN** code uses a stable feature introduced after 1.85
- **THEN** the MSRV job fails while the remaining pinned-toolchain jobs also evidence the incompatibility

### Requirement: Apache-2.0 license present and declared
The repository SHALL contain the full Apache-2.0 license text and the workspace manifests SHALL declare it as the project license.

#### Scenario: License visible in the repository
- **WHEN** the repository is inspected
- **THEN** the Apache-2.0 license file exists at the root and the license declaration appears in the workspace manifests

