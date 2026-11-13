# Spec Delta

## Purpose

Defines ONNX export: the `SKToOnnx` contract, graph construction, opset and dtype mapping,
per-family operators, metadata and validation.

## ADDED Requirements

### Requirement: ONNX export contract

A fitted model or pipeline SHALL expose an ONNX export method returning a valid ONNX model
byte sequence and its opset version.

#### Scenario: Export returns bytes

- **WHEN** a supported fitted model is exported
- **THEN** the result is a non-empty ONNX model with a declared opset

#### Scenario: Unsupported model is explicit

- **WHEN** a model family has no exporter
- **THEN** the call returns `UnsupportedOperation` with the family name

### Requirement: Graph builder

The exporter SHALL build a graph with typed inputs, initializers for learned tensors, nodes
for the model computation and typed outputs.

#### Scenario: Inputs and outputs are typed

- **WHEN** a model is exported
- **THEN** the graph declares the feature input shape/dtype and the prediction output
  shape/dtype

#### Scenario: Learned tensors become initializers

- **WHEN** a model has coefficients, centers or tree arrays
- **THEN** they are serialized as graph initializers, not constants in nodes

### Requirement: Opset and dtype mapping

The exporter SHALL use a documented ONNX opset and SHALL map `f32`/`f64` to the matching
ONNX tensor dtype.

#### Scenario: f32 export

- **WHEN** an `f32` model is exported
- **THEN** all floating inputs, initializers and outputs are `FLOAT`

#### Scenario: f64 export

- **WHEN** an `f64` model is exported
- **THEN** all floating inputs, initializers and outputs are `DOUBLE`

### Requirement: Per-family operators

The exporter SHALL cover the documented model families with ONNX/ONNX-ML operators and
SHALL record the supported coverage in the crate documentation.

#### Scenario: Linear model export

- **WHEN** a linear or logistic model is exported
- **THEN** the graph uses MatMul/Add plus the link operator (none/Sigmoid/Softmax)

#### Scenario: Tree ensemble export

- **WHEN** a tree or forest is exported
- **THEN** the graph uses the ONNX-ML tree-ensemble operator with the fitted thresholds
  and leaf values

#### Scenario: Pipeline export

- **WHEN** a pipeline is exported
- **THEN** the stages are composed in graph order and the final estimator produces the
  outputs

### Requirement: Metadata

The ONNX model SHALL carry producer metadata identifying the library, estimator, format
version and any class table needed to interpret the outputs.

#### Scenario: Classifier metadata

- **WHEN** a classifier is exported
- **THEN** the class table is present in the model metadata

### Requirement: Validation

The exporter SHALL reject empty states, unsupported dtypes, invalid shapes and missing
required tensors before producing a graph; exported graphs SHALL be loadable by a
compatible runtime.

#### Scenario: Exported graph loads

- **WHEN** a supported model is exported
- **THEN** a compatible ONNX runtime loads the graph and predicts the fixture within
  tolerance

#### Scenario: Missing tensor is rejected

- **WHEN** a required learned tensor is absent from the state
- **THEN** export returns `ShapeMismatch` or `UnsupportedOperation` naming it
