# Design

## Context

See `proposal.md` — Why. `sciencekit_math` owns pairwise distances and the shared
`SKDataView`; `sciencekit_common` owns builders, errors, execution mode and observability;
`sciencekit_metrics` exposes the evaluation functions the acceptance tests use. Linear
models already share a coordinate-descent substrate, but no crate implements a kernel
method. `sparse-pairwise-distances` (in flight) provides the mixed-representation distance
dispatcher used by the RBF kernel. There is no `memmap2` integration and no interop layer.

## Goals / Non-Goals

**Goals:**
- Kernel functions in `sciencekit_math` with dense/sparse/mixed parity.
- A deterministic pure-Rust SMO solver shared by classification and regression.
- Fitted models that store only support vectors and dual coefficients.
- Calibrated probabilities for the classifier and a clear convergence contract.

**Non-Goals:**
- `SKLinearSVC` (W3.6), nu-SVC/nu-SVR and one-class SVM.
- `memmap2`-backed kernel caches, persistence and model export (Wave 7).
- GPU, Python bindings and approximate kernels.
- Streaming/out-of-core SVM fitting: SMO needs random access to the kernel rows, so the
  first version is in-memory with a bounded cache.

## Decisions

### Decision 1 — Kernels belong to `sciencekit_math`

Linear, polynomial, RBF and sigmoid kernels live in a new `kernels` module beside
`pairwise`; they are numeric primitives reused by SVM, kernel PCA and kernel ridge. The
module exposes `sk_*_kernel_matrix` functions over `SKDataView` and a closed
`SKKernel` configuration enum. Precomputed kernels are a validation wrapper: the SVM
consumes the supplied square matrix directly.

Alternatives considered: kernels inside `sciencekit_svm` (rejected — repeats the
pairwise-distance dependency inversion and blocks reuse); a trait object per kernel
(rejected — closed catalogue, dynamic dispatch in the hot loop).

### Decision 2 — Kernel formulas reuse the pairwise kernels

- Linear: `X·Yᵀ`, using the sparse dot product when either side is sparse.
- Polynomial: `(gamma · linear + coef0)^degree`.
- RBF: `exp(-gamma · squared_euclidean)`, reusing the dense/sparse/mixed squared-distance
  dispatcher and clamping tiny negative values to zero.
- Sigmoid: `tanh(gamma · linear + coef0)`.

Every kernel dispatches on representation once, returns a dense matrix and never
densifies a sparse input. `gamma` defaults are resolved at fit time: `Scale` is
`1/(n_features · variance(X))`, `Auto` is `1/n_features`, and an explicit value is used
as supplied. The resolved scalar is stored in the fitted model.

### Decision 3 — SMO for both C-SVC and epsilon-SVR

The solver minimizes the standard dual with a second-order working-set selection
(Fan, Chen & Lin 2005), the maximal KKT violation as the stopping criterion, clipping to
`[0, C]` and a bias term recovered from the free support vectors. SVR uses the `2n`
formulation with `alpha`/`alpha*` and the epsilon-insensitive box. Both share the gradient
update, the working-set selection and the cache.

Alternatives considered: a generic QP solver from `faer` (no dedicated QP API and much
slower for the typical n); stochastic subgradient (simpler but a different algorithm from
the Wave plan, which locks Platt/LIBSVM-style SMO).

### Decision 4 — Bounded in-memory kernel cache

The solver stores kernel rows in an LRU cache keyed by training index, capped by
`cache_size` in MiB. A row is computed on demand from the kernel function over the stored
training representation (dense or CSR). When the cache cannot hold a row, it is recomputed
rather than spilled. `memmap2` spilling is explicitly deferred to Wave 7.

Alternatives considered: materializing the full `n × n` matrix (unbounded memory);
recomputing every row with no cache (too slow); memmap2 now (out of scope and not yet a
dependency).

### Decision 5 — One-vs-one multiclass with deterministic tie-breaking

For `c` classes the classifier fits `c(c−1)/2` binary models, each on the rows of its two
classes. Prediction counts votes; `decision_function` exposes the pairwise decision values
aggregated into a per-class confidence matrix. A vote tie is broken by the smallest
canonical class index, and the class table is the `SKLabelTable` order already used by the
other classifiers. Pairwise fits are independent and run in parallel with rayon.

Alternatives considered: one-vs-rest (cheaper but not the SVC formulation and a different
decision surface); a single multiclass QP (not the locked algorithm).

### Decision 6 — Probability calibration: Platt scaling plus pairwise coupling

Binary models compute the decision value and fit Platt parameters `A`/`B` by regularized
maximum likelihood on five deterministic cross-validation folds (Lin, Lin & Weng 2007).
Multiclass probabilities are obtained by pairwise coupling (Wu, Lin & Weng 2004) from the
binary probabilities, with a documented tolerance and iteration cap. Calibration is
enabled by the `probability` builder flag (default `false`, matching scikit-learn's cost
profile); when disabled, `predict_probabilities` returns `UnsupportedOperation` with
guidance to enable it.

Alternatives considered: always calibrating (unexpected fit cost); softmax over votes
(uncalibrated and not comparable with scikit-learn).

### Decision 7 — Model state: support vectors, dual coefficients, kernel configuration

`SKSVCModel<F>` stores the class table, one binary model per pair (support vectors, dual
coefficients, bias), the kernel configuration with resolved `gamma`, and the backend kind.
`SKSVRModel<F>` stores its support vectors, coefficients, bias, kernel configuration and
backend. Support vectors are owned in the training representation (`SKOwnedFeatureBlock`
with dense and CSR variants), so the model outlives the input and stays `Send + Sync`.
Prediction computes kernel rows against the support vectors and reduces them; it never
needs the original training matrix.

Alternatives considered: borrowing the training data with a lifetime (would infect the
model type and break the fitted-model contract); storing indices into an external matrix
(the caller could drop it).

### Decision 8 — Representation parity

Fit accepts dense, sparse or mixed features; the kernel and distance dispatchers handle
each cell. Prediction accepts dense, sparse or mixed query views against support vectors
stored densely or sparsely. No path densifies a sparse input. Precomputed kernels are
dense-only and reject a sparse matrix.

### Decision 9 — Determinism and iteration contract

The solver starts from zero dual coefficients and uses no randomness, so a fixed
configuration and input produce identical models. Probability calibration is the only
random consumer; it uses a seeded generator exposed as `random_seed` (default `0`).
`NotConverged { iterations }` is returned when the budget is exhausted; the model is not
returned partially fitted.

### Decision 10 — File layout

```
crates/sciencekit_math/src/kernels/
  mod.rs, linear.rs, polynomial.rs, radial_basis.rs, sigmoid.rs, config.rs, *_tests.rs
crates/sciencekit_svm/src/
  lib.rs
  kernel/{mod.rs, resolved.rs, distance.rs, *_tests.rs}
  smo/{mod.rs, working_set.rs, gradient.rs, cache.rs, stopping.rs, *_tests.rs}
  svc/{mod.rs, builder.rs, core_implementation.rs, model.rs, fit.rs, predict.rs, *_tests.rs}
  svr/{mod.rs, builder.rs, core_implementation.rs, model.rs, fit.rs, predict.rs, *_tests.rs}
  probability/{mod.rs, platt.rs, coupling.rs, *_tests.rs}
  acceptance_tests/mod.rs
```

`mod.rs` files are pure dispatchers; implementation files stay under 200 lines and
companion `*_tests.rs` modules are declared at the end.

## Risks / Trade-offs

- **`O(n²)` kernel work and `O(n_sv · n_features)` model state** → bounded cache and
  support-vector-only storage; document the profile.
- **SMO can be slow on large, noisy data** → second-order working set, shrinking and a
  configurable tolerance; `NotConverged` is explicit rather than silent.
- **Probability calibration multiplies fit cost** → default `false`; documented.
- **Polynomial kernel overflow for large degree / values** → document the numerical range;
  validation rejects non-finite inputs and tests cover the supported range.
- **Sparse kernel cache stores dense rows** → a cached row is `O(n)` regardless of input
  density; the cache is capped and sparse inputs are never densified as a whole.
- **Precomputed kernel shape** → validated as square and matched to the training-row
  count before the solver runs.
- **Backend is carried but the SMO solver is pure Rust** → the backend is recorded in
  spans and transferred to the model, preserving the uniform selection surface; the
  distance path may dispatch through the backend.

## Migration Plan

Additive: a new crate, a new math module and workspace membership. No existing signature
changes. Removing the change deletes the crate and the kernel module; `SKLinearSVC`
(W3.6) can reuse the kernel module independently.

## Open Questions

None blocking. `memmap2` cache spilling and model export are Wave 7; GPU and Python
bindings are separate accelerator changes.
