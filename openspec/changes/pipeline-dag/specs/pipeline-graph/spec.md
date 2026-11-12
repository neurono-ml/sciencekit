# Spec Delta

## Purpose

Defines the runtime DAG pipeline: graph construction, validation, topological fitting and
prediction over the shared feature-block medium.

## ADDED Requirements

### Requirement: DAG construction and validation

`SKPipelineGraph` SHALL be a directed acyclic graph of stages; validation SHALL reject
cycles, missing final estimators, unreachable nodes and multiple final estimators.

#### Scenario: Cycle is rejected

- **WHEN** a graph contains a cycle
- **THEN** validation returns `InvalidHyperparameter` naming the cycle

#### Scenario: One final estimator is required

- **WHEN** a graph has zero or two final estimator nodes
- **THEN** validation returns `InvalidHyperparameter`

#### Scenario: Disconnected node is rejected

- **WHEN** a node cannot be reached from the inputs
- **THEN** validation returns `InvalidHyperparameter`

### Requirement: Topological fit and predict

Fit SHALL execute nodes in topological order, fitting each transformer on its incoming
block and passing the transformed block to its successors; predict SHALL reuse the fitted
graph.

#### Scenario: Diamond graph

- **WHEN** one input branches to two transformers and merges into a final estimator
- **THEN** both branches are fitted and their outputs merge in the documented order

#### Scenario: Prediction matches a hand-built linear equivalent

- **WHEN** a diamond graph is equivalent to a linear chain
- **THEN** both produce the same predictions

### Requirement: Compatibility validation

Every edge SHALL carry an `SKFeatureBlock` and SHALL be validated for scalar compatibility
and representation acceptance before fitting.

#### Scenario: Scalar mismatch is rejected

- **WHEN** two nodes use different scalars
- **THEN** the graph cannot be constructed with both

#### Scenario: Representation rejection is surfaced

- **WHEN** a node rejects the incoming representation
- **THEN** the error names the node and representation

### Requirement: Deterministic execution

Topological execution SHALL be deterministic: nodes at the same level execute in a stable
node-id order, and results are stored by node id.

#### Scenario: Repeated runs agree

- **WHEN** the same graph is fitted twice with the same data
- **THEN** node outputs and final predictions are identical

### Requirement: Accessors and concurrency

The fitted graph SHALL expose node count, edge count, topological order, fitted node blocks
and the final model, and SHALL be safely shareable.

#### Scenario: Accessors describe the graph

- **WHEN** a graph is fitted
- **THEN** the topological order covers every node exactly once

#### Scenario: Concurrent prediction agrees

- **WHEN** several threads predict with the same fitted graph
- **THEN** every result equals the sequential result
