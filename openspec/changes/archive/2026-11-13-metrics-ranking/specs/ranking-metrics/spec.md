# Spec Delta

## Purpose

Defines the probability-based classification metrics: ROC/AUC, precision-recall/average
precision, log loss, balanced accuracy and Matthews correlation, their scorer adapters and
the AUC accumulator.

## ADDED Requirements

### Requirement: Probability scorer contract

The library SHALL expose a probability scorer with a pure form over stored class
probabilities and a convenient form that obtains probabilities from the model.

#### Scenario: Pure form does not re-infer

- **WHEN** stored probabilities and true targets are scored
- **THEN** no model inference runs

#### Scenario: Convenient form delegates

- **WHEN** a model is scored through the convenient form
- **THEN** the result equals the pure form over its `predict_probabilities` output

### Requirement: ROC curve and AUC

The library SHALL compute binary ROC curves and AUC from scores and labels, and multiclass
ROC/AUC with the one-vs-rest strategy; one-vs-one curves are a documented non-goal.

#### Scenario: Perfect ranking has unit AUC

- **WHEN** every positive score exceeds every negative score
- **THEN** AUC is `1`

#### Scenario: Random ranking has half AUC

- **WHEN** positives and negatives are interleaved uniformly
- **THEN** AUC is approximately `0.5`

#### Scenario: Ties are handled deterministically

- **WHEN** several scores are equal
- **THEN** the ROC/AUC computation uses a documented tie rule and repeated runs agree

#### Scenario: Multiclass averaging

- **WHEN** a multiclass problem is scored with macro, micro or weighted averaging
- **THEN** the result equals the documented aggregate of the per-class AUCs

### Requirement: Precision-recall curve and average precision

The library SHALL compute precision-recall curves and average precision from scores and
labels, with the documented step interpolation.

#### Scenario: Average precision of a perfect ranking

- **WHEN** all positives rank before all negatives
- **THEN** average precision is `1`

#### Scenario: Empty positive class is rejected

- **WHEN** the labels contain no positive class
- **THEN** the metric returns a structured error

### Requirement: Log loss

The library SHALL compute binary and multiclass log loss from probabilities, clipping
probabilities to the documented epsilon to avoid infinities.

#### Scenario: Confident correct prediction has low loss

- **WHEN** the true class probability is near one
- **THEN** the log loss is near zero

#### Scenario: Confident wrong prediction is finite

- **WHEN** the true class probability is zero
- **THEN** the loss is finite because of clipping, never infinity

### Requirement: Balanced accuracy and Matthews correlation

The library SHALL compute balanced accuracy as the mean per-class recall and Matthews
correlation coefficient from labels.

#### Scenario: Balanced classes

- **WHEN** classes are balanced and all labels are correct
- **THEN** balanced accuracy and MCC are `1`

#### Scenario: Majority-only prediction

- **WHEN** a classifier predicts only the majority class on an imbalanced problem
- **THEN** balanced accuracy is `0.5` and MCC is `0`

### Requirement: Metric scorer adapters

The library SHALL provide scorer adapters for AUC, average precision, log loss, balanced
accuracy and MCC implementing the probability or label scorer contracts.

#### Scenario: Scorer equals free function

- **WHEN** a scorer adapter and the corresponding free function evaluate the same data
- **THEN** both return the same value within tolerance

### Requirement: AUC accumulator

The library SHALL provide a mergeable accumulator for AUC that accepts score/label pairs
and produces the same AUC as the whole-input function; its O(n) memory profile SHALL be
documented.

#### Scenario: Batched equals whole-input

- **WHEN** the same pairs are accumulated in batches and scored at once
- **THEN** both AUCs are identical

#### Scenario: Merge equals concatenation

- **WHEN** two accumulators are merged
- **THEN** the result equals one accumulator over the concatenated pairs

### Requirement: Validation and concurrency

The metrics SHALL reject empty input, length mismatch, invalid score ranges, single-class
AUC and non-finite values, and SHALL be deterministic under concurrent calls.

#### Scenario: Length mismatch is rejected

- **WHEN** scores and labels have different lengths
- **THEN** the metric returns `ShapeMismatch`

#### Scenario: Concurrent calls agree

- **WHEN** several threads compute the same metric
- **THEN** every result equals the single-threaded result
