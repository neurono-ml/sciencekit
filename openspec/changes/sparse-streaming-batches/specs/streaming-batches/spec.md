# Spec Delta

## MODIFIED Requirements

### Requirement: Owned batches with minimal metadata
Each streaming block SHALL fully own its data (no borrowing from the source), carrying its position in the sequence and final-block indication, so it can be moved to another thread while the source advances.

#### Scenario: Block survives the source
- **WHEN** a block is extracted from the source's iterator and the reference to the iterator is dropped right after
- **THEN** the block's data remains intact and usable — the property enabling asynchronous processing off the reading thread

#### Scenario: Last block is identifiable
- **WHEN** a finite source is consumed to the end
- **THEN** exactly one block is marked as final

#### Scenario: A sparse block is moved like a dense one
- **WHEN** a block carrying a CSR matrix is moved to another thread
- **THEN** its sparse representation and metadata remain intact, with no conversion

### Requirement: Supervised streaming source

The library SHALL define a supervised streaming contract whose owned blocks pair a feature matrix with its targets, so a supervised algorithm fits out-of-core without materializing the whole dataset. The block SHALL fully own its data and carry its position and a final-block indication; the source SHALL expose itself as fallible block iteration with errors from the central taxonomy.

#### Scenario: Supervised block owns features and targets

- **WHEN** a supervised block is extracted from the source and the reference to the
  source is dropped right after
- **THEN** the block's features and targets remain intact and usable

#### Scenario: Supervised source as fallible iteration

- **WHEN** reading an intermediate supervised block fails
- **THEN** iteration yields the central taxonomy error and the consumer decides
  whether to stop or handle it

#### Scenario: Features may be sparse

- **WHEN** a supervised block carries a CSR feature matrix
- **THEN** its targets pair with the block's rows and the representation reaches the
  estimator unchanged
