## Purpose

Custom global allocator selection: `allocator-jemalloc` and `allocator-mimalloc` feature
flags that swap in `tikv-jemallocator` or `mimalloc` respectively, with the system allocator
as the default.

## ADDED Requirements

### Requirement: Allocator is selectable by feature flag
The library SHALL select the global allocator through feature flags: `allocator-jemalloc`
installs `tikv-jemallocator`, `allocator-mimalloc` installs `mimalloc`, and neither enabled
uses the system allocator.

#### Scenario: Default build uses the system allocator
- **WHEN** the crate is built without an allocator feature
- **THEN** the global allocator is the system allocator

#### Scenario: Jemalloc feature installs jemalloc
- **WHEN** the `allocator-jemalloc` feature is enabled
- **THEN** the global allocator is `tikv-jemallocator`

#### Scenario: Mimalloc feature installs mimalloc
- **WHEN** the `allocator-mimalloc` feature is enabled
- **THEN** the global allocator is `mimalloc`

### Requirement: Conflicting allocator features are rejected
The library SHALL reject a build that enables both `allocator-jemalloc` and
`allocator-mimalloc` at once, since a single global allocator can be installed at most once.

#### Scenario: Both allocator features conflict
- **WHEN** both `allocator-jemalloc` and `allocator-mimalloc` are enabled together
- **THEN** the build fails with a clear conflict error