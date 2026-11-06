# Design

## Context

See `proposal.md` — Why. `sciencekit_tree` (W4.1) grows CART structures with a response
and computes leaf values from its criterion. Boosting needs the structure grown on
pseudo-residuals but leaf values computed from gradients and Hessians, so the tree crate
must expose a structure-only build plus leaf replacement. `sciencekit_ensemble` owns the
boosting substrate shared with the histogram variant.

## Goals / Non-Goals

**Goals:**
- A loss-agnostic stage loop reusable by the histogram booster.
- Correct Newton leaf values and shrinkage.
- Binary and multiclass classification plus regression.
- Early stopping without a separate model family.

**Non-Goals:**
- Histogram binning (W4.5), categorical features, missing-value routing.
- Streaming/`partial_fit`, `memmap2` and export.
- Quantile and Poisson losses.

## Decisions

### Decision 1 — Boosting substrate in `sciencekit_ensemble`

A shared module defines `SKGradientLoss<F>`:

```
initial_prediction(targets) -> F (or a vector for multiclass)
negative_gradient(target, prediction) -> F
hessian(target, prediction) -> F
link / inverse link
loss(prediction, target) -> F
```

Classification uses the log-loss with a sigmoid (binary) or softmax (multiclass) link;
regression implements squared error, absolute error and Huber. The stage loop consumes
the trait, so the tree code and the histogram booster share it.

Alternative considered: a separate loop per loss (duplicated early stopping, shrinkage and
multiclass handling).

### Decision 2 — Structure-only tree build plus leaf replacement

The tree crate gains an advanced surface:

```
SKTreeStructure::build_structure(features, pseudo_residuals, constraints) -> structure
structure.replace_leaf_values(values: &[F]) -> Result<(), SKError>
```

The public `SKDecisionTreeRegressor` keeps its criterion-driven leaf values and does not
use this surface. Boosting builds the structure on the negative gradient with the tree's
squared-error split criterion, then replaces each leaf value with the Newton step. The
advanced surface validates the leaf count and leaves the structure unchanged on error.

### Decision 3 — Newton leaf values and shrinkage

For a leaf with gradient sum `G` and Hessian sum `H`:

```
value = -G / (H + lambda)
```

and the model adds `learning_rate * value` to the raw score. For squared error this equals
the leaf mean of the residuals; for log loss it is the Newton step. Leaves with `H = 0`
(degenerate) keep value zero.

### Decision 4 — Stage loop

For each stage:

1. compute the negative gradient and Hessian for every training row from the current raw
   score;
2. optionally subsample rows (`subsample < 1`) with the seeded generator;
3. grow a structure on the selected rows with `maximum_depth`,
   `minimum_samples_leaf` and `maximum_features`;
4. replace leaf values with the Newton step and scale by `learning_rate`;
5. add the tree to the model and update the raw scores.

Binary classification grows one tree per stage; multiclass grows one tree per class per
stage and updates each raw score with its class tree. Tree fitting is single-stage
sequential by definition; split search inside a tree is feature-parallel.

### Decision 5 — Initial prediction

Regression starts at the target mean. Binary classification starts at the log-odds of the
positive prior. Multiclass starts at the class log-priors. The initial prediction is stored
on the model and added before the stage trees.

### Decision 6 — Early stopping

When `n_iter_no_change` is set, a seeded `validation_fraction` split is held out before
fitting. After each stage the validation loss is computed; the best stage count and its
model state are retained, and fitting stops after `n_iter_no_change` non-improving stages.
The model reports the fitted stage count.

### Decision 7 — Prediction surface

Regression: `init + Σ learning_rate · tree(x)`. Binary: sigmoid of that score gives the
positive probability. Multiclass: softmax over the per-class scores. Labels use the
canonical class table and the smallest-index tie-break. Feature importances aggregate the
stage trees weighted by sample count.

### Decision 8 — Representation parity

The tree base learner accepts dense and CSR. Boosting materializes only the row subset it
fits (a row copy, not a densification) and passes it through. Prediction dispatches once
and uses the tree's dense/sparse traversal.

### Decision 9 — Validation and observability

Builder defaults and validation follow the proposal. `fit` transfers backend/execution
mode to the model; every span records the backend. Errors: `EmptyInput`,
`InvalidHyperparameter`, `ShapeMismatch`, `NonFiniteValue`, `NotConverged` (not used),
and the tree error converted into the central taxonomy.

### Decision 10 — File layout

```
crates/sciencekit_tree/src/structure/
  mod.rs, build.rs, leaf_values.rs, *_tests.rs
crates/sciencekit_ensemble/src/boosting/
  mod.rs, loss/{mod.rs, squared.rs, absolute.rs, huber.rs, log_loss.rs, *_tests.rs},
  stage.rs, early_stopping.rs, init.rs, *_tests.rs
crates/sciencekit_ensemble/src/gradient_boosting/
  mod.rs, classifier.rs, regressor.rs, builder.rs, model.rs, fit.rs, predict.rs, *_tests.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines.

## Risks / Trade-offs

- **Absolute-error Hessian is degenerate** → use a constant Hessian `1` and document the
  approximation; the leaf median is not the same as the Newton step, so tests assert the
  documented behavior.
- **Multiclass uses a diagonal Hessian** → standard one-tree-per-class approximation;
  documented.
- **Early stopping needs a validation split** → seeded and reproducible; disabled when
  `n_iter_no_change` is unset.
- **Leaf replacement is an advanced API** → covered by the decision-tree delta and tests;
  public CART behavior is unchanged.
- **Stage fitting is memory-resident** → streaming is a later change.

## Migration Plan

Additive: new ensemble estimators and an advanced tree surface. No existing API changes.

## Open Questions

None blocking. Categorical features and missing routing arrive with the histogram variant.
