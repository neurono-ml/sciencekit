## ADDED Requirements

### Requirement: Supervised streaming source

The library SHALL define a supervised streaming contract whose owned blocks pair
a feature matrix with its targets, so a supervised algorithm fits out-of-core
without materializing the whole dataset. The block SHALL fully own its data and
carry its position and a final-block indication; the source SHALL expose itself
as fallible block iteration with errors from the central taxonomy.

#### Scenario: Supervised block owns features and targets

- **WHEN** a supervised block is extracted from the source and the reference to
  the source is dropped right after
- **THEN** the block's features and targets remain intact and usable

#### Scenario: Supervised source as fallible iteration

- **WHEN** reading an intermediate supervised block fails
- **THEN** iteration yields the central taxonomy error and the consumer decides
  whether to stop or handle it

### Requirement: The prefetch driver serves both source kinds

The streaming executor SHALL drive both the feature-only and the supervised
sequential sources through the same prefetch pipeline, overlapping I/O with
compute under the plan's buffer depth and delivering blocks in source order.

#### Scenario: Supervised driver pairs features and targets

- **WHEN** the supervised prefetch driver runs over a supervised source
- **THEN** each update step receives that block's features and targets, in
  source order
