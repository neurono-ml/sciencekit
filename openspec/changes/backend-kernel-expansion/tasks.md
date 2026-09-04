## 1. Modularize backend/ (pure-dispatcher mod.rs)

- [ ] 1.1 Create `backend/kernel.rs` defining `SKMathBackend<F>` and `SKNormKind`
- [ ] 1.2 Create `backend/decompositions.rs` with concrete `SKSingularValueDecomposition<F>`, `SKQRDecomposition<F>`, `SKLUDecomposition<F>`
- [ ] 1.3 Create `backend/dispatch.rs` with `sk_default_math_backend<F>()`
- [ ] 1.4 Rewrite `backend/mod.rs` as a pure dispatcher: declarations and re-exports only
- [ ] 1.5 Update `sciencekit_math/src/lib.rs` re-exports for the new module layout

## 2. Genericize the trait over SKFloat (TDD)

- [ ] 2.1 Red: write companion tests expecting a generic `F: SKFloat` surface (both `f32` and `f64`)
- [ ] 2.2 Make `SKMathBackend` generic over `F: SKFloat`; replace hard-coded `f64` in every signature
- [ ] 2.3 Implement `SKMathBackend<F>` for `SKFaerBackend` and `SKMatrixMultiplyBackend` over `f32`/`f64`
- [ ] 2.4 Implement `SKMathBackend<F>` for `SKNdArrayLinalgBackend` (feature-gated) over `f32`/`f64`
- [ ] 2.5 Add an `f32` reconstruction test (SVD/QR) alongside the `f64` one; keep suite green

## 3. Host-centric ndarray surface (remove faer from the public surface)

- [ ] 3.1 Red: add a compile-time assertion that the trait surface exposes no `faer` types
- [ ] 3.2 Change trait inputs to `ArrayView2<F>` and outputs to `Array2<F>`
- [ ] 3.3 Convert decomposition structs to hold `Array2<F>`; LU pivot as `Vec<usize>`
- [ ] 3.4 Move the faer↔ndarray conversion inside each backend's boundary (zero-copy on the public side)
- [ ] 3.5 Rewrite `backend_tests.rs` to use ndarray fixtures (`ndarray`/`sprs`, no `faer` in tests)

## 4. Expanded kernel operations (TDD, one primitive at a time)

- [ ] 4.1 `solve_triangular` (upper/lower) — red, minimal impl, green
- [ ] 4.2 `solve` (general `Ax = b`)
- [ ] 4.3 `eigh` (symmetric eigendecomposition, eigenvalues + eigenvectors)
- [ ] 4.4 `lu` with `Vec<usize>` pivot; verify `P A = L U` reconstruction
- [ ] 4.5 `slogdet` returning `(sign, log_abs_det)`; verify against a known near-singular matrix (no underflow)
- [ ] 4.6 `pinv` (Moore–Penrose) via SVD
- [ ] 4.7 `inv` for square matrices
- [ ] 4.8 `lstsq` (minimum-norm least squares, rank-deficient stable) — see design Decision 7
- [ ] 4.9 `norm`: matrix + vector, full `ord` set (`Frobenius`, `L2`, `L1`, `Infinity`, negatives, `Nuclear`, `General`) with specialized + general fallback paths

## 5. Internal parallelism honours the execution plan

- [ ] 5.1 Red: parallel GEMM test expecting dispatch across a configured `parallelism`
- [ ] 5.2 Thread `parallelism` through `gemm` (faer `Par`, BLAS thread level) instead of `Par::Seq`
- [ ] 5.3 Green: large GEMM scales with the plan's parallelism without changing results

## 6. Acceptance (PRD §8.7)

- [ ] 6.1 Verify the full kernel on both small and large data sizes
- [ ] 6.2 Verify `Send + Sync` and safe operation under concurrency for every backend
- [ ] 6.3 Verify backend `kind()` is recorded and the model/result is exportable across backends

## 7. Out-of-core roadmap marker

- [ ] 7.1 Record the `svds`/`eigsh` out-of-core follow-up (tracking reference) and confirm the trait surface does not preclude adding truncated decompositions later