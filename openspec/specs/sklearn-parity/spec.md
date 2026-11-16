# sklearn-parity Specification

## Purpose
Make the scikit-learn parity claim falsifiable. This capability owns the pinned
reference surface, the exhaustive status matrix of every estimator, transformer and
function family, the declared memory-access tier of each entry, and the mechanical
cross-checks that keep the matrix, the implementation and the change graph from
drifting apart.

## Requirements

### Requirement: Pinned scikit-learn baseline

The parity surface MUST be pinned to one scikit-learn release, and the pinned API listing
MUST be stored in the repository. The matrix MUST NOT be derived from a floating "latest"
release at check time.

#### Scenario: Baseline is recorded

- **WHEN** the matrix is read
- **THEN** it names exactly one baseline version and one reference listing file
- **AND** every entry can be traced to a line of that listing

#### Scenario: Baseline upgrade is an explicit change

- **WHEN** a newer scikit-learn release is adopted
- **THEN** the reference listing and the affected matrix rows change in the same commit
- **AND** new entries arrive as `planned` or `absent`, never silently as `implemented`

### Requirement: Exhaustive single-status matrix

Every public estimator and transformer class of the baseline MUST appear exactly once in
the matrix, with exactly one status: `implemented`, `planned` or `absent`. Function
families (metrics, dataset generators, inspection utilities, preprocessing functions)
MUST also appear, grouped at the family granularity defined by the matrix schema.

#### Scenario: Duplicate or missing entry is rejected

- **WHEN** the checker reads the matrix
- **THEN** an entry that appears twice fails the check
- **AND** a baseline class with no entry fails the check

#### Scenario: Status vocabulary is closed

- **WHEN** an entry declares a status outside `implemented`, `planned`, `absent`
- **THEN** the check fails with the offending entry named

### Requirement: Access-pattern tier per entry

Every entry whose status is `implemented` or `planned` MUST declare one access-pattern
tier: `streaming`, `mappable`, `chunked` or `full-memory`. An `absent` entry MAY omit the
tier, because no implementation is promised. The tier MUST match the doctrine of
the `access-pattern-contract` capability: a `streaming` or `mappable` estimator whose
implementation does not consume the corresponding source kind is not `implemented`.

#### Scenario: Tier and implementation agree

- **WHEN** an entry is `implemented` with tier `mappable`
- **THEN** its change's completion evidence includes a test that fits the estimator from
  an `SKMappableSource`
- **AND** an entry whose tier is `full-memory` names the documented alternative path

#### Scenario: Missing tier is rejected

- **WHEN** an implemented or planned entry has no tier
- **THEN** the check fails

### Requirement: Planned entries map to the change graph

Every `planned` entry MUST name an owning change that exists in
`openspec/changes/implementation-dag/dag.yaml`, and that change MUST be reachable from
the graph's lanes.

#### Scenario: Dangling owner is rejected

- **WHEN** a planned entry names a change absent from `dag.yaml`
- **THEN** the check fails and lists both the matrix row and the missing change id

#### Scenario: Absent entry explains itself

- **WHEN** an entry is `absent`
- **THEN** it carries a non-empty reason
- **AND** an absent entry MUST NOT name an owning change

### Requirement: Coverage cannot silently regress

The matrix MUST support a coverage report (counts per status and per family) and the
repository MUST publish that report in the documentation. A change that implements a
matrix entry MUST flip its status in the same PR that lands the implementation.

#### Scenario: Implementation without matrix update is caught

- **WHEN** a fitted model exists for a matrix entry still marked `planned`
- **THEN** the consistency check reports the stale row
- **AND** the change cannot be marked complete

#### Scenario: Removed implementation is caught

- **WHEN** an entry marked `implemented` loses its coverage test
- **THEN** the check fails instead of silently counting it
