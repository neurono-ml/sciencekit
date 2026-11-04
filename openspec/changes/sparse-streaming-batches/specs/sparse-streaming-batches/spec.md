## Purpose

Defines the owned feature block carried by streaming batches: a dense matrix or a CSR matrix
with representation-aware accessors, so an out-of-core fit can stream sparse data without
densifying it and without changing the source contract.

## ADDED Requirements

### Requirement: An owned block carries one of two representations

The library SHALL define an owned feature block with exactly two representations — a dense
row-major matrix and a CSR matrix — SHALL be non-exhaustive for future representations, and
SHALL guarantee that the block fully owns its data.

#### Scenario: Dense block owns its matrix

- **WHEN** a dense block is created and its source is dropped
- **THEN** the block's data remains intact and usable

#### Scenario: Sparse block owns its CSR arrays

- **WHEN** a sparse block is created and its source is dropped
- **THEN** the block's CSR arrays remain intact and usable

### Requirement: Accessors expose the representation without copying

A block SHALL expose its row and column counts, whether it is sparse, its estimated resident
byte size, and a zero-copy view over either representation. Obtaining the view SHALL not
copy, densify or convert the data.

#### Scenario: A sparse view borrows the block

- **WHEN** the view of a sparse block is requested
- **THEN** the returned view borrows the block's CSR arrays and no dense matrix is allocated

#### Scenario: Shape is reported for both representations

- **WHEN** a block is built from a known matrix
- **THEN** its row and column counts equal the matrix dimensions, whichever representation it
  carries

### Requirement: Byte size reflects the stored representation

The block SHALL report its resident byte size from its actual buffers: `rows × columns ×
size_of(scalar)` for a dense block, and the element, index and pointer arrays for a sparse
block. The execution plan's memory guard SHALL use that size.

#### Scenario: Sparse size follows the non-zeros

- **WHEN** a sparse block with few non-zeros is measured
- **THEN** its byte size is far below the dense `rows × columns × size_of(scalar)` figure

#### Scenario: The guard measures the stored buffers

- **WHEN** a streaming plan resolves over a sparse batch hint
- **THEN** the doubled size checked against available memory is the sparse size, not the
  dense size

### Requirement: Streaming batches carry either representation

The feature-only and supervised batches SHALL carry a feature block, SHALL preserve their
position and final-block flag, and SHALL be constructible from either representation. A
supervised batch SHALL pair the block with targets of the same row count and SHALL reject a
mismatch with a shape error.

#### Scenario: A sparse supervised batch pairs targets

- **WHEN** a supervised batch is built from a sparse block and one target per row
- **THEN** it exposes the sparse view and the targets together

#### Scenario: Row counts must agree

- **WHEN** a supervised batch is built with a target count different from the block's rows
- **THEN** construction returns a shape error
