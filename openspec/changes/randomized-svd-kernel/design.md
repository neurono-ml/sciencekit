# Design

## Context

See `proposal.md` — Why. `sciencekit_math::backend` exposes full `svd` and `eigh` over
`faer`; `sprs` provides CSR products; there is no randomized or truncated SVD. The
wave-plan dependency candidates (`rsvd-faer`, `faer-sparse`) are incompatible with the
workspace's `faer 0.24`, so this change implements the algorithm directly.

## Goals / Non-Goals

**Goals:**
- A deterministic randomized SVD over dense and sparse matrices.
- Accuracy good enough for PCA/TruncatedSVD reconstruction.
- No new linear-algebra stack and no densification of sparse input.

**Non-Goals:**
- ARPACK/IRLBA, full SVD (already available), GPU.
- Mixed dense/sparse inputs (a one-matrix kernel has one representation).
- An adaptive stopping criterion (power iterations are caller-configured).

## Decisions

### Decision 1 — Hand-rolled randomized SVD on the existing backend

The algorithm:

```
Omega = seeded_random(n, k + oversampling)
Y = X * Omega
repeat power_iterations:
    Q = qr(Y).q
    Z = X^T * Q
    Y = X * Z
Q = qr(Y).q
B = Q^T * X
(U_B, s, V) = svd(B)
U = Q * U_B
truncate to k components
```

Dense products use the backend `gemm`; sparse products use `sprs` row/column scans and
never materialize a dense `X`. `Q` is dense (`m × (k+p)`), which is small and expected.

Alternative considered: depending on `rsvd-faer` (faer 0.19 duplicate stack) and
`faer-sparse` (different release line); rejected after verifying the crate metadata.

### Decision 2 — Seeded random projection without a new dependency

A small SplitMix64-based generator produces a Rademacher or Gaussian projection matrix from
`random_seed`. The generator is local to the module; a future shared `SKRandomGenerator`
in `sciencekit_common` can replace it without changing the public API. The projection is
generated in a deterministic order, so the same seed always yields the same `Omega`.

### Decision 3 — Orthonormalization and power iterations

Each power iteration applies `X (Xᵀ Y)` and re-orthonormalizes `Y` with QR to avoid
collapse. The final `B = Qᵀ X` is small (`(k+p) × n`) and is decomposed with the backend's
full SVD; `U` is lifted back with `Q`. The result is truncated to `k` components.

### Decision 4 — Deterministic sign convention

Singular vectors are defined up to sign. After the final decomposition, each component's
sign is fixed by making the largest-magnitude entry of `U` positive; an exact tie is
resolved by the smallest row index. The corresponding column of `V` is flipped with it, so
the factorization stays consistent.

### Decision 5 — Module and API

```
crates/sciencekit_math/src/decomposition/
  mod.rs
  randomized.rs
  randomized_tests.rs
```

Public surface:

```rust
pub struct SKRandomizedSvdResult<F: SKFloat> {
    pub u: Array2<F>,
    pub singular_values: Vec<F>,
    pub v: Array2<F>,
}

pub fn sk_randomized_svd<F: SKFloat>(
    matrix: SKDataView<'_, F>,
    number_of_components: usize,
    number_of_power_iterations: usize,
    oversampling: usize,
    random_seed: u64,
) -> Result<SKRandomizedSvdResult<F>, SKError>;
```

### Decision 6 — Validation and errors

Validate non-empty input, `k <= min(m, n)`, `power_iterations >= 1` and
`oversampling >= 1`; return `InvalidHyperparameter`/`EmptyInput` before computing.
`number_of_components == 0` is rejected explicitly rather than producing an empty result.

## Risks / Trade-offs

- **Randomized approximation error** → power iterations improve it; tests assert a
  documented reconstruction tolerance on low-rank and decaying-spectrum fixtures.
- **Sparse products can be slower than dense** → the algorithm never densifies; the sparse
  path is the point.
- **Large `k + oversampling` can exceed the matrix rank** → the final SVD returns fewer
  components; the result truncates to `min(k, rank)` and documents it.
- **Sign convention can flip between implementations** → fixed deterministically in this
  code and tested across repeated calls.
- **No adaptive convergence** → callers choose power iterations; a future adaptive variant
  can be added without breaking the API.

## Migration Plan

Additive: a new module and free function. No existing API changes.

## Open Questions

None blocking. An adaptive/ARPACK variant is a future accelerator.
