# Spec Delta

## Purpose

Defines ONNX import as a generic predictor: parsing, input/output mapping, supported
operators, classifier/regressor adaptation and validation.

## ADDED Requirements

### Requirement: Generic ONNX predictor

The library SHALL load an ONNX model and expose a generic predictor over dense feature
views, without converting the model into a native estimator type.

#### Scenario: Regressor output

- **WHEN** an imported model produces a single numeric output per row
- **THEN** the predictor returns one response per row

#### Scenario: Classifier output

- **WHEN** an imported model produces a class-probability matrix
- **THEN** the predictor exposes probabilities and argmax labels

### Requirement: Input mapping

The predictor SHALL accept a dense feature matrix with the model's declared input width
and dtype, and SHALL reject a mismatch before inference.

#### Scenario: Feature-count mismatch

- **WHEN** the input width differs from the model's declared input width
- **THEN** the predictor returns `ShapeMismatch`

#### Scenario: Dtype mismatch

- **WHEN** the model expects `f32` and receives `f64`
- **THEN** the predictor either converts explicitly or returns a structured error, never
  silently reinterprets bytes

### Requirement: Supported operators

The importer SHALL document the supported operator set and SHALL reject an unsupported
operator or model construct with `UnsupportedOperation` naming it.

#### Scenario: Supported graph runs

- **WHEN** a model uses only supported operators
- **THEN** inference completes and returns finite outputs

#### Scenario: Unsupported operator

- **WHEN** a model uses an operator outside the supported set
- **THEN** loading or inference returns `UnsupportedOperation` with the operator name

#### Scenario: Malformed graph structure

- **WHEN** the graph's data dependencies form a cycle, a node declares too few
  operands, or shape metadata disagrees with a learned tensor
- **THEN** loading or inference returns a typed `SKOnnxImportError` naming the
  construct and the library never panics

### Requirement: Batch behavior

Inference SHALL preserve the input row count and SHALL support batches of arbitrary size
within the model's declared batch semantics.

#### Scenario: Batch of one

- **WHEN** a single row is predicted
- **THEN** the output has one row

#### Scenario: Large batch

- **WHEN** a large batch is predicted
- **THEN** the output row count matches the input and values agree with row-by-row
  inference

### Requirement: Validation and concurrency

The importer SHALL reject truncated/invalid model bytes, missing inputs and multiple
ambiguous outputs, and a loaded predictor SHALL be safely shareable.

#### Scenario: Invalid bytes

- **WHEN** the input is not a valid ONNX model
- **THEN** loading returns a structured conversion error

#### Scenario: Concurrent prediction agrees

- **WHEN** several threads predict with the same imported model
- **THEN** every result equals the sequential result
