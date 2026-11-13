# native-model-import Specification

## Purpose
Defines native model import: which imported models convert into native types, how the
conversion validates state/graphs and when continued training is supported.

## Requirements

### Requirement: Safetensors native import

A supported Safetensors model state SHALL convert into the matching native fitted model
using `SKImportable` and the documented tensor names.

#### Scenario: Linear model round-trip

- **WHEN** a Safetensors state for a linear model is imported natively
- **THEN** the native model predicts the same values as the original within tolerance

#### Scenario: Unsupported family

- **WHEN** a state names a family without a native importer
- **THEN** conversion returns `UnsupportedOperation`

### Requirement: ONNX native import

The library SHALL convert a documented subset of ONNX graphs (linear/preprocessing/tree
ensembles) into native models by mapping initializers and operators to native state.

#### Scenario: Supported graph

- **WHEN** an ONNX graph uses only the documented subset
- **THEN** the native model predicts the same values as the generic ONNX predictor

#### Scenario: Unsupported operator

- **WHEN** the graph uses an operator outside the subset
- **THEN** conversion returns `UnsupportedOperation` naming it

### Requirement: Continued training

For models whose native configuration and state are reconstructible, the imported model
SHALL support continued fitting (warm start or `partial_fit`) with the documented semantics.

#### Scenario: Warm-start linear model

- **WHEN** an imported linear model is fitted on additional data
- **THEN** fitting continues from the imported coefficients and produces a valid model

#### Scenario: Unsupported retraining

- **WHEN** a family cannot be retrained from its imported state
- **THEN** the attempt returns `UnsupportedOperation` with guidance

### Requirement: Validation and concurrency

Conversion SHALL validate estimator name/version, feature count, tensor shapes/dtypes and
class tables before constructing a model; conversions SHALL be deterministic and safe under
concurrency.

#### Scenario: Shape mismatch

- **WHEN** a state tensor has the wrong shape
- **THEN** conversion returns `ShapeMismatch` and no model

#### Scenario: Concurrent conversion

- **WHEN** several threads convert the same state
- **THEN** every native model is identical
