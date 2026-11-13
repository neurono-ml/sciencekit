# Spec Delta

## Purpose

Defines the calibration metrics (curves with uniform and quantile strategies, top-label
reliability data, Brier score and expected calibration error) and the multilabel metrics
(Hamming loss, Jaccard score, subset accuracy, precision/recall/F1 with the four averaging
modes and label-ranking average precision) over dense and CSR indicator matrices.

## ADDED Requirements

### Requirement: Calibration curve with two binning strategies

The library SHALL compute a calibration curve from binary targets and positive-class
probabilities using either a uniform strategy, with a configurable number of equal-width
bins over `[0, 1]`, or a quantile strategy, with a configurable number of empirical
quantile bins.

#### Scenario: Uniform bins have equal width

- **WHEN** a curve is computed with the uniform strategy and ten bins
- **THEN** consecutive bin bounds differ by `0.1` and the first bin includes `0`

#### Scenario: Quantile bins hold equal counts

- **WHEN** a curve is computed with the quantile strategy on distinct probabilities
- **THEN** every non-empty bin contains an approximately equal number of samples

#### Scenario: Duplicate quantile edges collapse

- **WHEN** many probabilities are identical and quantile edges coincide
- **THEN** the curve collapses the duplicate edges and returns fewer bins, never an empty
  width-zero bin

#### Scenario: Empty uniform bins are retained

- **WHEN** a uniform bin receives no sample
- **THEN** it is present with a zero sample count and zero summaries, so the curve length
  equals the configured bin count

### Requirement: Typed calibration result

The calibration result SHALL be a typed curve value composed of typed bins, each exposing
its lower and upper bound, sample count, fraction of positives and mean predicted
probability, together with the strategy and total sample count.

#### Scenario: Bin summaries are typed fields

- **WHEN** a caller inspects a calibration bin
- **THEN** every summary is a named field, never an element of a parallel tuple vector

#### Scenario: Fractions are consistent

- **WHEN** a bin contains samples
- **THEN** its fraction of positives and mean predicted probability both lie in `[0, 1]`

### Requirement: Top-label reliability data

The library SHALL derive top-label reliability data from a multiclass probability matrix
and integer targets, using the argmax class as the predicted label and its probability as
the confidence, and returning the same typed calibration curve.

#### Scenario: Confidence is the maximum probability

- **WHEN** a multiclass probability row is reduced to top-label data
- **THEN** its confidence equals the row maximum

#### Scenario: Correctness follows the argmax

- **WHEN** the argmax class equals the true class
- **THEN** the row contributes a positive outcome to the curve

#### Scenario: Argmax ties are deterministic

- **WHEN** two classes share the maximum probability
- **THEN** the smallest class index is selected and repeated runs agree

### Requirement: Brier score

The library SHALL compute the binary Brier score as the mean squared difference between
the positive-class probability and the binary outcome.

#### Scenario: Perfect probabilities score zero

- **WHEN** every probability equals its outcome
- **THEN** the Brier score is `0`

#### Scenario: Constant half probability scores one quarter

- **WHEN** every probability is `0.5`
- **THEN** the Brier score is `0.25`

#### Scenario: Non-binary probability shape is rejected

- **WHEN** the probability matrix does not have exactly two columns
- **THEN** the function returns `ShapeMismatch`

### Requirement: Expected calibration error

The library SHALL compute the top-label expected calibration error over a configurable
number of bins and binning strategy, as the sample-weighted mean absolute difference
between per-bin accuracy and confidence.

#### Scenario: Perfectly calibrated confidence has zero error

- **WHEN** every bin's accuracy equals its mean confidence
- **THEN** the expected calibration error is `0`

#### Scenario: Error is bounded

- **WHEN** any valid probabilities are scored
- **THEN** the expected calibration error lies in `[0, 1]`

#### Scenario: Invalid bin count is rejected

- **WHEN** the configured number of bins is zero
- **THEN** the function returns `InvalidHyperparameter`

### Requirement: Multilabel representation parity

Every multilabel metric SHALL accept a dense indicator matrix and a CSR indicator matrix,
dispatch on the representation exactly once per call and never densify a sparse input.

#### Scenario: Dense and sparse agree

- **WHEN** the same logical indicator data is scored densely and as CSR
- **THEN** every metric agrees within the documented tolerance

#### Scenario: Sparse input is not densified

- **WHEN** a CSR indicator matrix is scored
- **THEN** only stored entries are read and no dense row copy is allocated

#### Scenario: Non-indicator values are rejected

- **WHEN** an indicator matrix contains a value other than zero or one outside tolerance
- **THEN** the function returns `InvalidTarget` naming the offending row and column

### Requirement: Hamming loss and subset accuracy

The library SHALL compute the multilabel Hamming loss as the fraction of incorrectly
predicted labels and the subset accuracy as the fraction of rows whose predicted set equals
the true set.

#### Scenario: Exact prediction has zero loss

- **WHEN** every predicted label set equals the true set
- **THEN** the Hamming loss is `0` and the subset accuracy is `1`

#### Scenario: One wrong label counts once

- **WHEN** exactly one entry of an indicator matrix is flipped
- **THEN** the Hamming loss equals `1 / (rows · labels)`

#### Scenario: Partially correct rows fail subset accuracy

- **WHEN** a row has one missing and one extra label
- **THEN** the Hamming loss is positive and the row does not count toward subset accuracy

### Requirement: Jaccard score and averaging

The library SHALL compute the multilabel Jaccard score with binary, micro, macro, weighted
and samples averaging, documenting the zero-division policy.

#### Scenario: Perfect sets score one

- **WHEN** predicted and true sets are identical
- **THEN** the Jaccard score is `1` for every averaging mode

#### Scenario: Disjoint sets score zero

- **WHEN** predicted and true sets share no label
- **THEN** the Jaccard score is `0`

#### Scenario: Zero denominators contribute zero

- **WHEN** a label has no true and no predicted positives
- **THEN** its per-label score is `0` under the documented policy and the aggregate matches
  the scikit-learn value

### Requirement: Multilabel precision, recall and F1 with four averaging modes

The library SHALL compute per-label and aggregate multilabel precision, recall and F1 with
micro, macro, weighted and samples averaging, returning a typed report with per-label
scores and support.

#### Scenario: Micro pools the counts

- **WHEN** micro averaging is requested
- **THEN** precision, recall and F1 are computed from the pooled true-positive,
  false-positive and false-negative counts

#### Scenario: Macro is the unweighted label mean

- **WHEN** macro averaging is requested
- **THEN** each metric equals the arithmetic mean of its per-label values

#### Scenario: Weighted uses support

- **WHEN** weighted averaging is requested
- **THEN** each per-label value is weighted by its support and the result equals the
  scikit-learn value

#### Scenario: Samples averages rows

- **WHEN** samples averaging is requested
- **THEN** the metric is computed per row and averaged over rows

#### Scenario: Per-label report is complete

- **WHEN** any averaging mode is requested
- **THEN** the report exposes precision, recall, F1 and support for every label

### Requirement: Label-ranking average precision

The library SHALL compute multilabel label-ranking average precision from predicted scores
and a true indicator matrix, averaging per-row precision at the ranks of true labels.

#### Scenario: Perfect ranking scores one

- **WHEN** every true label outranks every false label in every row
- **THEN** label-ranking average precision is `1`

#### Scenario: A row without true labels is rejected

- **WHEN** any row of the true indicator matrix has no positive label
- **THEN** the function returns `InvalidTarget`

### Requirement: Scikit-learn parity documentation

Every calibration and multilabel metric SHALL document its scikit-learn counterpart, its
averaging semantics, its zero-division policy and its dense/sparse equivalence tolerance.

#### Scenario: Counterpart is named

- **WHEN** a caller reads a metric's documentation
- **THEN** the corresponding scikit-learn function or the documented non-reference
  definition is named

#### Scenario: Averaging semantics are explicit

- **WHEN** a metric accepts an averaging mode
- **THEN** the documentation states what each mode aggregates and how zero denominators are
  treated

### Requirement: Validation and concurrency

The metric functions SHALL reject empty input, shape or length mismatch, invalid
probabilities, invalid bin counts, invalid strategies and non-finite values, and SHALL be
safe to call concurrently.

#### Scenario: Empty input is rejected

- **WHEN** a metric receives zero rows
- **THEN** it returns `EmptyInput`

#### Scenario: Length mismatch is rejected

- **WHEN** the target length differs from the probability or indicator row count
- **THEN** the metric returns `ShapeMismatch`

#### Scenario: Non-finite probability is rejected

- **WHEN** a probability is NaN or infinite
- **THEN** the metric returns `NonFiniteValue`

#### Scenario: Concurrent calls agree

- **WHEN** several threads run the same metric on the same data
- **THEN** every result equals the sequential result
