# FDR — Feature & Example Design Records

An **FDR** is the design record of one example: the complete recipe of what the example
builds, from the dataset to the last hyperparameter, together with the behavior the
reader must observe and the sciencekit capability it needs.

The scikit-learn gallery gives the *shape* of each example — which dataset, which
estimator, which knob to turn. It does not give the *recipe*: scikit-learn's pages
assume a reader who already knows what to select, what to engineer and what to chain.
A sciencekit FDR spells the whole pipeline out, because the corpus is also the teaching
material of the machine-learning book.

## Why the record is complete

Every FDR answers eight questions in order. An example that cannot answer all of them
is not ready to be implemented:

| Stage | Question | Why it matters |
|---|---|---|
| **Dataset** | What data, from where, of what shape? | The dataset's properties (scale, missingness, cardinality) decide which preprocessing is meaningful. |
| **Feature selection** | Which columns reach the estimator, and why? | Selection leaks if fitted before the split; the FDR states the order explicitly. |
| **Feature engineering** | What is derived from the raw columns? | Derived features are where most of the signal — and most of the leakage — is created. |
| **Preprocessing** | What transformations, fitted where? | Every scaler and imputer is fitted on the training split only, and the FDR says so. |
| **Pipeline** | What is the ordered chain of stages? | The chain *is* the example; the reader must be able to read it left to right. |
| **Spectrum** | Which settings are compared? | An example that shows one configuration teaches nothing about the effect of configuration. |
| **Effect** | What must the reader observe, and why? | The cause-and-effect the book chapter will teach. |
| **Output** | Which metrics and which figure? | What the companion test asserts and what the page shows. |

## The pipeline grammar

```
ColumnSelection → SimpleImputer → OneHotEncoder → StandardScaler → PolynomialFeatures → Estimator
```

- **Stages are ordered, and the order is part of the example.** Selection before
  imputation would drop the rows that need imputing.
- **Every fitted stage is fitted on the training split only** and then applied to the
  test split; the FDR says so once per record, and every FDR assumes it.
- **A stage that is not in the chain is a decision**: "none — the raw columns are
  already scaled" is a valid and meaningful answer.
- The chain names sciencekit types (`SKStandardScaler`, `SKRidge`), not scikit-learn
  ones; the mapping to the scikit-learn original is in the **Origin** line.

## Conventions

- **Identifiers** are unabbreviated English, snake_case, matching the example id, the
  module name and the cargo example target — the single exception being the canonical
  mathematical symbol of an algorithm (`k`, `t`, `p`).
- **Availability** is one of: `available now`, `planned (<crate>)`, `absent (no
  sciencekit path)`. A record marked `planned` is a promise the catalog's `blocked_by`
  field carries; a record marked `absent` is a deliberate omission with a reason.
- **Seeds** are explicit in every record that touches a stochastic component; the
  companion test asserts the resulting numbers.
- **Numbers** are quoted with a documented precision and asserted with a documented
  tolerance, so a page is stable across platforms and math backends.
- Each record keeps the scikit-learn origin visible: the gallery page it adapts, its
  file path and its title. The page carries a credit line; the record carries the
  provenance.

## The record

```markdown
### `example_id`

**Origin** — scikit-learn `examples/<family>/plot_<name>.py` · *"<title>"*

| Stage | Recipe |
|---|---|
| Dataset | … |
| Feature selection | … |
| Feature engineering | … |
| Preprocessing | … |
| Pipeline | … |
| Spectrum | … |
| Effect | … |
| Output | … |
| Availability | … |
```

## Files

| File | Covers |
|---|---|
| `availability-matrix.md` | every FDR id, its status and the crate that owns it |
| `01-preprocessing-and-imputation.md` | scalers, encoders, polynomial expansion, imputation |
| `02-linear-models.md` | least squares, penalized regression, logistic regression, SGD |
| `03-neighbors-and-svm.md` | k-nearest neighbours, support vector machines |
| `04-trees-and-ensembles.md` | decision trees, bagging, boosting, forests |
| `05-clustering.md` | k-means, mini-batch k-means, DBSCAN |
| `06-decomposition.md` | PCA, incremental PCA, truncated SVD |
| `07-model-selection-and-pipelines.md` | cross-validation, grid search, halving, pipelines |
| `08-metrics-and-inspection.md` | confusion matrix, ROC, PR curves, calibration, permutation importance |
| `09-feature-selection-and-engineering.md` | selection strategies and derived-feature recipes |
| `10-outliers-covariance-and-long-tail.md` | anomaly detection, covariance, and the families sciencekit does not cover yet |

## Relationship to the catalog

The FDR is the **design** of an example; the catalog record is its **state**. The FDR
is written once, by the designer; the catalog record is filled in as the example moves
from `blocked` to `ported`, and it carries the fields that change over time — the
status, the blocker, the languages, the pointer to the implementation.
