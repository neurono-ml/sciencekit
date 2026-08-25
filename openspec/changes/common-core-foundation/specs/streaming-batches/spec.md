## Purpose

Defines the PRD out-of-core streaming contract (§4.4): owned data blocks with minimal metadata, transferable across threads enabling the I/O ∥ CPU pipeline, and the two complementary sources — sequential by iteration and abstract random access (memmap arrives at interop).

## ADDED Requirements

### Requirement: Owned batches with minimal metadata
Each streaming block SHALL fully own its data (no borrowing from the source), carrying its position in the sequence and final-block indication, so it can be moved to another thread while the source advances.

#### Scenario: Block survives the source
- **WHEN** a block is extracted from the source's iterator and the reference to the iterator is dropped right after
- **THEN** the block's data remains intact and usable — the property enabling asynchronous processing off the reading thread

#### Scenario: Last block is identifiable
- **WHEN** a finite source is consumed to the end
- **THEN** exactly one block is marked as final

### Requirement: Sequential source as a fallible iterator
The sequential streaming source SHALL expose itself as block iteration with errors from the central taxonomy, allowing intermediate read failures without panicking.

#### Scenario: Read failure stops with a structured error
- **WHEN** reading an intermediate block fails
- **THEN** iteration yields the central taxonomy error and the consumer decides whether to stop or handle it

### Requirement: Abstract random-access source
The library SHALL define the contract of a source with direct positional access to data units, independent of the storage mechanism; concrete memory-mapped implementations belong to the interop layer.

#### Scenario: Direct positional access without scanning
- **WHEN** an arbitrary unit is requested by index from a random-access source
- **THEN** access does not require traversing previous units

#### Scenario: Contract does not couple storage mechanism
- **WHEN** a provider implements the source over any own persistence mechanism
- **THEN** no memory-mapping-specific dependency is required by the contract definition
