## Purpose

Defines the shared stochastic-gradient substrate of `sciencekit_linear_model`: the loss
catalogue, the learning-rate schedules, the proximal penalty update, per-sample weighting,
iterate averaging, seeded shuffling, the convergence and early-stopping bookkeeping, and the
single epoch driver that serves dense, sparse and streaming inputs.

## ADDED Requirements

### Requirement: The objective is the weighted mean loss plus a penalty

The substrate SHALL minimize the mean over samples of the loss of the linear predictor
`f(x) = xᵀw + b` scaled by that sample's weight, plus the penalty `P(w)`. It SHALL accept
`alpha ≥ 0` and `l1_ratio ∈ [0, 1]`; `P(w)` SHALL be `0.5·alpha·‖w‖₂²` (`L2`),
`alpha·‖w‖₁` (`L1`), `alpha·(l1_ratio·‖w‖₁ + 0.5·(1−l1_ratio)·‖w‖₂²)` (`ElasticNet`) or `0`
(`None`). The intercept SHALL never be penalized. `l1_ratio` SHALL only be consulted under
`ElasticNet`.

#### Scenario: Zero alpha disables regularization

- **WHEN** an estimator is fitted with `alpha(0.0)`
- **THEN** the fit succeeds and the coefficients are not shrunk

#### Scenario: Growing alpha shrinks the coefficients

- **WHEN** the same data is fitted with an increasing `alpha`
- **THEN** the L2 coefficient magnitude decreases monotonically

#### Scenario: L1 produces exact zeros

- **WHEN** a design with irrelevant features is fitted with an L1 penalty and enough epochs
- **THEN** the irrelevant coefficients are exactly zero

#### Scenario: The intercept is not penalized

- **WHEN** data whose best fit has a large intercept is fitted with growing `alpha`
- **THEN** the intercept stays near its unpenalized estimate while the slopes shrink

### Requirement: Invalid hyperparameters are rejected

The substrate SHALL reject a negative `alpha`, an `l1_ratio` outside `[0, 1]`, a
non-positive `tolerance`, a zero `maximum_number_of_iterations`, a non-positive initial
learning rate under a schedule that uses it, a non-positive inverse-scaling power, a
`validation_fraction` outside `(0, 1)`, and a zero
`number_of_epochs_without_improvement`, each with `SKError::InvalidHyperparameter` naming
the field.

#### Scenario: Negative alpha is rejected

- **WHEN** an estimator is built with a negative `alpha`
- **THEN** the build returns an `InvalidHyperparameter` error naming `alpha`

#### Scenario: Invalid validation fraction is rejected

- **WHEN** early stopping is enabled with a `validation_fraction` of one
- **THEN** the build returns an `InvalidHyperparameter` error naming the field

#### Scenario: A non-positive initial rate is rejected when used

- **WHEN** `Constant`, `InverseScaling` or `Adaptive` is selected with an initial learning
  rate of zero
- **THEN** the build returns an `InvalidHyperparameter` error naming the initial rate

### Requirement: The loss catalogue is split by task family

The substrate SHALL provide classifier losses `hinge`, `log_loss`, `modified_huber`,
`squared_hinge` and `perceptron`, and regressor losses `squared_error`, `huber`,
`epsilon_insensitive` and `squared_epsilon_insensitive`. Each loss SHALL expose its value,
its subgradient with respect to the linear predictor, and whether it yields calibrated
probabilities. A loss from the other family SHALL be rejected at build with a structured
error.

#### Scenario: Probabilistic losses are identified

- **WHEN** the classifier loss is `log_loss` or `modified_huber`
- **THEN** the fitted model reports that probabilities are available

#### Scenario: Non-probabilistic losses are identified

- **WHEN** the classifier loss is `hinge`, `squared_hinge` or `perceptron`
- **THEN** the fitted model reports that probabilities are unavailable

#### Scenario: The insensitive losses honor epsilon

- **WHEN** a residual falls inside the configured `epsilon` band
- **THEN** it contributes no loss and no gradient

### Requirement: Learning-rate schedules are deterministic and selectable

The substrate SHALL support the `Constant`, `Optimal`, `InverseScaling` and `Adaptive`
learning-rate schedules and SHALL maintain a monotone update counter shared by `fit`,
`partial_fit` and `fit_streaming`.

#### Scenario: Constant keeps the rate

- **WHEN** the same data is fitted with `Constant`
- **THEN** every update uses the configured initial learning rate

#### Scenario: Inverse scaling decays

- **WHEN** the same data is fitted with `InverseScaling` and a positive power
- **THEN** the effective step size decreases monotonically with the update counter

#### Scenario: Optimal follows the inverse-strength form

- **WHEN** `Optimal` is selected with a positive regularization strength
- **THEN** the step size follows `1 / (alpha · (step + t₀))` for a fixed positive `t₀`, and
  the initial learning rate is not consulted

#### Scenario: Optimal requires a positive strength

- **WHEN** `Optimal` is selected with `alpha` of zero
- **THEN** the build returns an `InvalidHyperparameter` error naming the combination

#### Scenario: Adaptive halves on a plateau

- **WHEN** an epoch fails to improve the objective under `Adaptive`
- **THEN** the next epoch uses half the previous learning rate and never a larger one

#### Scenario: The counter continues across incremental calls

- **WHEN** `partial_fit` follows `fit`
- **THEN** the learning rate reflects the total number of updates, not just the new batch's

### Requirement: Class weights are expanded per sample

The substrate SHALL accept the shared class-weighting policy and expand it into one
non-negative weight per sample, leaving the penalty unweighted. `Balanced` SHALL weight each
class by `n_samples / (n_classes · class_count)`. A negative weight, a wrong length or an
all-zero weighting SHALL be rejected with a structured error.

#### Scenario: Balanced weighting equalizes class totals

- **WHEN** an imbalanced design is fitted with `Balanced`
- **THEN** each class contributes equal total weight within tolerance

#### Scenario: Negative weight is rejected

- **WHEN** an explicit class weight is negative
- **THEN** the fit returns a structured error naming the class weight

### Requirement: Iterate averaging is optional

With averaging enabled the substrate SHALL report the running mean of the iterates visited
by the driver instead of the last iterate, updated incrementally without storing the
trajectory; with averaging disabled it SHALL report the last iterate. The averaged state
SHALL survive `partial_fit`.

#### Scenario: Averaging reports a different vector

- **WHEN** a noisy design is fitted with averaging enabled
- **THEN** the reported coefficients differ from the last iterate and are the running mean
  of the visited iterates

#### Scenario: Averaging continues incrementally

- **WHEN** `partial_fit` follows a fit with averaging enabled
- **THEN** the reported coefficients depend on the earlier updates, not only on the new batch

### Requirement: Shuffling is seeded and reproducible

The substrate SHALL support an optional per-epoch shuffle driven by a seed carried on the
builder. The same seed and the same data SHALL produce the same update order and the same
coefficients; different seeds SHALL produce different orders; disabling shuffling SHALL keep
the input order. Streaming SHALL shuffle within a batch only, because a one-shot source
cannot be rewound.

#### Scenario: Same seed, same result

- **WHEN** the same design is fitted twice with the same seed and shuffling enabled
- **THEN** the coefficient vectors are identical

#### Scenario: Different seeds differ

- **WHEN** the same design is fitted with two different seeds
- **THEN** the coefficient vectors differ

#### Scenario: Shuffling disabled keeps the order

- **WHEN** shuffling is disabled
- **THEN** the update order follows the input row order

### Requirement: Convergence, the budget and the policy are explicit

The substrate SHALL track the epoch objective and SHALL consider a fit converged when the
objective fails to improve by more than `tolerance` for
`number_of_epochs_without_improvement` consecutive epochs. The fitted model SHALL report
whether it converged and how many epochs it executed.

#### Scenario: A plateau converges before the budget

- **WHEN** a well-conditioned design is fitted with a generous budget
- **THEN** the fit converges and the model reports fewer epochs than the budget

#### Scenario: Strict policy rejects an exhausted budget

- **WHEN** the `Strict` policy is active and the full requested budget executes without
  meeting the plateau criterion
- **THEN** the fit returns a non-convergence error carrying the iterations executed

#### Scenario: Permissive policy returns the last iterate

- **WHEN** the `Permissive` policy is active and the budget executes without meeting the
  plateau criterion
- **THEN** the model is returned, reports that it did not converge, and a warning is emitted

#### Scenario: A provider-limited fit is not an exhausted budget

- **WHEN** a one-shot source supplies fewer passes than the requested budget
- **THEN** the model is returned reporting the passes taken and that it did not converge,
  even under the `Strict` policy

### Requirement: Early stopping uses a held-out validation slice

With early stopping enabled, the in-memory fit SHALL reserve a deterministic
`validation_fraction` slice and SHALL score the validation objective at the end of every
epoch, keeping the best-scoring iterate. Early stopping SHALL require more than one epoch
of budget.

#### Scenario: Early stopping keeps the best iterate

- **WHEN** the validation objective worsens after a few improving epochs
- **THEN** the reported coefficients are those of the best validation epoch and the fit
  stops after `number_of_epochs_without_improvement` non-improving epochs

#### Scenario: Early stopping needs more than one epoch

- **WHEN** early stopping is enabled with a budget of one epoch
- **THEN** the build returns an `InvalidHyperparameter` error

#### Scenario: Early stopping over a stream is refused

- **WHEN** `fit_streaming` is called with early stopping enabled
- **THEN** the fit returns an `UnsupportedOperation` error

### Requirement: One update rule serves dense, sparse and batch inputs

The epoch driver SHALL apply the same per-sample update over a dense row view, a borrowed
CSR row and a supervised streaming batch, SHALL never densify a sparse design, and SHALL
retain only the coefficients, the intercept, the schedule counter and the averaging state.
Under the same seed and update order, the dense and sparse fits SHALL agree within the
documented relative tolerance.

#### Scenario: Sparse equals dense under a fixed seed

- **WHEN** the same design is fitted densely and as an equivalent CSR matrix with the same
  seed and shuffling disabled
- **THEN** the coefficients agree within tolerance

#### Scenario: A sparse fit never densifies

- **WHEN** a borrowed CSR design is fitted
- **THEN** no dense `n × p` copy of the design is constructed

#### Scenario: Streaming retains nothing but the model state

- **WHEN** a source yields many batches
- **THEN** the peak retained data is one batch plus the coefficient state
