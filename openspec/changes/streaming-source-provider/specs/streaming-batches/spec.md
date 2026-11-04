# Spec Delta

## MODIFIED Requirements

### Requirement: The prefetch driver serves both source kinds

The streaming executor SHALL drive both the feature-only and the supervised sequential sources through the same prefetch pipeline, overlapping I/O with compute under the plan's buffer depth and delivering blocks in source order.

#### Scenario: Supervised driver pairs features and targets

- **WHEN** the supervised prefetch driver runs over a supervised source
- **THEN** each update step receives that block's features and targets, in
  source order

#### Scenario: Each pass opens its own source

- **WHEN** the streaming executor drives a reopenable provider with a budget above one
- **THEN** it opens a source for each pass and runs the prefetch pipeline over that source,
  preserving block order within the pass

#### Scenario: A one-shot provider is not reopened

- **WHEN** the streaming executor drives a one-shot provider
- **THEN** it opens the source once and never requests a further pass
