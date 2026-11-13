# decision-trees Specification

## Purpose
Defines the CART decision-tree classifier and regressor: the builder surface, impurity
and split-search contract, tree structure and accessors, prediction and probability
outputs, dense/sparse representation parity, determinism and validation.

## Requirements

### Requirement: Builder-only construction with documented defaults

Both estimators SHALL be constructible only through their builders, exposing `criterion`,
`maximum_depth`, `minimum_samples_split` (default 2), `minimum_samples_leaf` (default 1),
`maximum_features` (`All` default), `splitter` (`Best` default), `random_seed` (default
0), `class_weight` (classifier only, `None` default), `backend` (default `Faer`) and
`execution_mode` (default `Automatic`).

#### Scenario: Default configuration

- **WHEN** a tree is built without calling any setter
- **THEN** it grows without a depth limit, splits with at least two samples, leaves at
  least one sample per leaf, considers every feature, uses the best splitter, carries
  `Faer` and `Automatic`, and uses the estimator-specific default criterion

#### Scenario: Backend reaches the fitted model

- **WHEN** a non-default backend is configured and the tree is fitted
- **THEN** the fitted model records that backend in its prediction spans

### Requirement: CART split search

The tree SHALL grow by binary recursive partitioning, choosing at each node the split
that maximizes the impurity decrease for the configured criterion, subject to the
minimum-samples and depth constraints.

#### Scenario: A known split is recovered

- **WHEN** a two-class data set has a single feature that perfectly separates the classes
- **THEN** the root split is placed between the two classes and both leaves are pure

#### Scenario: Constraint prevents an invalid split

- **WHEN** a candidate split would leave fewer samples than `minimum_samples_leaf`
- **THEN** that split is not selected and the node either finds another split or becomes
  a leaf

#### Scenario: Depth limit stops growth

- **WHEN** `maximum_depth` is reached
- **THEN** the node becomes a leaf even if further impurity decrease is possible

### Requirement: Classification criteria and probabilities

The classifier SHALL support Gini and entropy impurity, SHALL predict the majority class
at each leaf, and SHALL expose class probabilities from the leaf class distribution.

#### Scenario: Gini and entropy both fit

- **WHEN** the same data is fitted with `Gini` and with `Entropy`
- **THEN** both models produce finite predictions and valid impurity values

#### Scenario: Leaf probabilities sum to one

- **WHEN** probabilities are requested for a batch
- **THEN** each row sums to one and contains one entry per fitted class

#### Scenario: Majority class is predicted

- **WHEN** a leaf holds two class-0 samples and one class-1 sample
- **THEN** rows reaching that leaf are predicted as class `0`

### Requirement: Regression criteria and prediction

The regressor SHALL support squared-error and absolute-error criteria, SHALL predict the
leaf mean for squared error and the leaf median for absolute error, and SHALL expose a
response per input row.

#### Scenario: Squared error predicts the leaf mean

- **WHEN** a leaf holds responses `1.0`, `2.0` and `6.0`
- **THEN** its prediction is `3.0`

#### Scenario: Absolute error predicts the leaf median

- **WHEN** a leaf holds responses `1.0`, `2.0` and `6.0`
- **THEN** its prediction is `2.0`

### Requirement: Feature subsampling

The tree SHALL support `All`, `Count`, `Fraction`, `Sqrt` and `Log2` feature
subsampling at each split, using the seeded generator for deterministic selection.

#### Scenario: Sqrt considers the square root of the features

- **WHEN** `maximum_features(Sqrt)` is configured on a nine-feature problem
- **THEN** each split considers three features

#### Scenario: Seeded selection is repeatable

- **WHEN** the same `random_seed` and data are fitted twice
- **THEN** the selected features and the resulting tree are identical

### Requirement: Feature importances

The model SHALL expose normalized feature importances derived from the impurity decrease
attributed to each feature, summing to one when the tree splits at least once.

#### Scenario: The separating feature dominates

- **WHEN** a tree is fitted on data where one feature separates the classes
- **THEN** that feature has the largest importance and the importances sum to one

### Requirement: Representation parity

Both trees SHALL accept dense and borrowed CSR features for fit and prediction, dispatch
once per operation and never densify a sparse design.

#### Scenario: Sparse fit equals dense fit

- **WHEN** the same logical design is fitted densely and as CSR
- **THEN** the split structure and predictions agree

#### Scenario: Sparse prediction is supported

- **WHEN** a model predicts on a sparse batch
- **THEN** prediction reads the stored entries and succeeds without densifying

### Requirement: Deterministic fitting and prediction

For a fixed configuration and input the tree SHALL be identical across repeated fits, and
tie-breaking between equally good splits SHALL be deterministic (smaller feature index,
then smaller threshold).

#### Scenario: Repeated fits agree

- **WHEN** the same tree fits the same data twice
- **THEN** node count, split thresholds and predictions are identical

#### Scenario: Parallel prediction equals sequential prediction

- **WHEN** a large batch is predicted through the parallel path and sequentially
- **THEN** both results are identical

### Requirement: Validation

The trees SHALL reject empty input, non-finite values, a feature-count mismatch, a
single-class classification target, invalid depth/split/leaf parameters and an invalid
feature-subsampling configuration with structured central errors.

#### Scenario: Empty input is rejected

- **WHEN** fit receives zero rows
- **THEN** it returns `EmptyInput` naming the operation

#### Scenario: Single-class classification is rejected

- **WHEN** the classifier receives only one class
- **THEN** fit returns `InvalidHyperparameter` naming the class requirement

#### Scenario: Non-finite value is rejected

- **WHEN** fit or prediction receives NaN or infinity
- **THEN** the operation returns `NonFiniteValue` naming the position and does not treat
  the value as missing

### Requirement: Concurrent prediction

Fitted trees SHALL be safely shareable across threads and SHALL produce the same result
under concurrent prediction as under a single thread.

#### Scenario: One model serves several threads

- **WHEN** several threads predict with the same fitted model
- **THEN** every call completes without locking and returns its sequential result

### Requirement: Structure-only tree build and leaf replacement

The tree crate SHALL expose an advanced surface that grows a tree structure from a
response vector without committing to the public estimator's leaf-value policy, and that
replaces every leaf value with a caller-supplied value while preserving the split
structure.

#### Scenario: Boosting replaces leaf values

- **WHEN** gradient boosting grows a structure on pseudo-residuals and computes Newton
  leaf values
- **THEN** the crate replaces every leaf value with the supplied value and the split
  structure is unchanged

#### Scenario: Leaf replacement validates length

- **WHEN** the supplied leaf values do not match the number of leaves
- **THEN** the operation returns `ShapeMismatch` and leaves the structure unchanged

#### Scenario: Public estimator behavior is unchanged

- **WHEN** `SKDecisionTreeRegressor` is fitted through its public builder
- **THEN** its leaf values follow its documented criterion and the advanced surface is not
  required
