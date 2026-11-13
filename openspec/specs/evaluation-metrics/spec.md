# evaluation-metrics Specification

## Purpose
Defines the library's concrete evaluation metrics for classification and regression, the
deterministic label-space and averaging policy for multiclass scores, the scorer adapters
over the shared scoring contracts, and mergeable streaming accumulators that produce the
same values as whole-input evaluation.

## Requirements

### Requirement: Deterministic classification label space

The library SHALL resolve the label space of a classification metric from an explicit
sorted label list when one is supplied, and otherwise from the sorted union of the observed
true and predicted labels. Every derived result SHALL order its classes by that resolution.

#### Scenario: Explicit labels fix the class order

- **WHEN** a metric is evaluated with an explicit label list `[2, 0, 1]`
- **THEN** the result orders its classes as `0, 1, 2`, independently of the observation order

#### Scenario: Default label space is the observed union

- **WHEN** no label list is supplied and the true labels contain `0` while the predictions contain `1`
- **THEN** both `0` and `1` belong to the resolved label space

#### Scenario: Observations outside an explicit label space are rejected

- **WHEN** an observation carries a label absent from the supplied explicit label list
- **THEN** the metric returns a structured error naming the offending label and never drops the observation silently

### Requirement: Confusion matrix

The library SHALL expose a confusion matrix whose rows are true labels and columns are
predicted labels, carrying unsigned counts over the resolved label space.

#### Scenario: Counts match the observations

- **WHEN** three observations have true label `0` and predicted labels `0, 1, 0`
- **THEN** the matrix holds two counts at true `0` / predicted `0` and one count at true `0` / predicted `1`

#### Scenario: Every resolved class gets a row and a column

- **WHEN** a class belongs to the resolved label space but never appears in the predictions
- **THEN** its column exists and holds zero counts

### Requirement: Accuracy

The library SHALL compute accuracy as the number of exactly matching label pairs divided
by the number of observations, and SHALL reject an empty observation set.

#### Scenario: Accuracy over a known example

- **WHEN** four observations have three correct predictions
- **THEN** accuracy is `0.75`

#### Scenario: Empty observations are rejected

- **WHEN** accuracy is computed over zero observations
- **THEN** a structured error is returned instead of a numeric sentinel

### Requirement: Per-class precision, recall and F1 with averaging

The library SHALL compute precision, recall and F1 per class and aggregate them with
macro, micro or weighted averaging. For a binary problem it SHALL also expose a
positive-label form reporting the metrics of one chosen class. F1 SHALL be the harmonic
mean of precision and recall.

#### Scenario: Binary positive-label form describes the chosen class

- **WHEN** a binary problem is scored with positive label `1`
- **THEN** the result equals the per-class precision, recall and F1 of class `1`

#### Scenario: Macro averaging weights classes equally

- **WHEN** a rare and a frequent class are scored with macro averaging
- **THEN** each class contributes with the same weight regardless of its support

#### Scenario: Micro averaging of single-label multiclass equals accuracy

- **WHEN** a single-label multiclass problem is scored with micro-averaged F1
- **THEN** the value equals the accuracy over the same observations

#### Scenario: Weighted averaging follows class support

- **WHEN** a class holds twice the support of another with weighted averaging
- **THEN** the larger class contributes twice the weight to the aggregate

### Requirement: Deterministic zero-division policy

The library SHALL define a single deterministic policy for a precision or recall whose
denominator is zero: the component is zero. Evaluation SHALL NOT produce NaN nor panic.

#### Scenario: Absent predicted class yields zero precision

- **WHEN** a class present in the true labels is never predicted
- **THEN** its precision is zero rather than NaN

#### Scenario: Absent true class yields zero recall

- **WHEN** a class is predicted but never present in the true labels
- **THEN** its recall is zero rather than NaN

### Requirement: Regression metrics

The library SHALL expose mean squared error, root mean squared error, mean absolute error
and the coefficient of determination over equal-length continuous targets and predictions.

#### Scenario: Perfect predictions

- **WHEN** predictions equal the targets
- **THEN** both errors are zero and the coefficient of determination is one

#### Scenario: Known error example

- **WHEN** targets are `[1, 2, 3]` and predictions are `[1, 2, 4]`
- **THEN** the mean squared error is `1/3` and the mean absolute error is `1/3`

#### Scenario: Length mismatch is rejected

- **WHEN** targets and predictions have different lengths
- **THEN** a structured shape error is returned

#### Scenario: Constant targets have a defined coefficient of determination

- **WHEN** every target is identical and the predictions are exact
- **THEN** the coefficient of determination is one; with a non-zero residual it is zero

### Requirement: Scorer adapters over the shared contracts

The library SHALL provide scorer types implementing `SKLabelScorer` and
`SKSupervisedScorer`, so any compatible fitted model is scored through the pure and the
convenient forms without additional adaptation.

#### Scenario: Pure form scores stored outputs without inference

- **WHEN** a scorer is evaluated with already-computed predictions
- **THEN** the metric is produced from those outputs with no model inference

#### Scenario: Convenient form matches the pure form

- **WHEN** the same model is scored through the convenient form and through the pure form applied to its predictions
- **THEN** both values are equal

### Requirement: Mergeable streaming accumulators

The library SHALL provide classification and regression accumulators that ingest batches,
merge with other accumulators, and produce the same metric values as whole-input
evaluation. Classification accumulation SHALL be bounded by the squared class count and
regression accumulation SHALL be constant-size.

#### Scenario: Batched accumulation equals whole-input evaluation

- **WHEN** observations arrive in several batches through one accumulator
- **THEN** the finalized metrics equal those of a single whole-input evaluation

#### Scenario: Merging two accumulators equals one concatenated pass

- **WHEN** two accumulators observe disjoint partitions and are merged
- **THEN** the finalized metrics equal one pass over the concatenation of both partitions

#### Scenario: Empty accumulator is rejected on finalization

- **WHEN** an accumulator that observed nothing is finalized
- **THEN** a structured error is returned instead of a numeric value

### Requirement: Concurrent accumulation is safe and deterministic

Accumulators SHALL be safe to share across compute threads and SHALL merge partitions
produced in parallel without additional synchronization from the consumer.

#### Scenario: Parallel partitions match the sequential result

- **WHEN** partitioned observations are accumulated in parallel and merged
- **THEN** the finalized metrics equal the sequential single-pass result

### Requirement: Non-finite observations are rejected

Classification and regression metrics SHALL reject non-finite targets or predictions with
a structured error rather than propagating NaN or infinity into the result.

#### Scenario: NaN prediction is rejected

- **WHEN** a prediction is NaN
- **THEN** the metric returns a structured error identifying the offending position
