## 1. Rule codification

- [x] 1.1 Add the "tests at the end" rule to `AGENTS.md` (inline `#[cfg(test)]` blocks at the end of a file; companion `*_tests.rs` declared at the end of `mod.rs`)
- [x] 1.2 Mirror the rule in the OpenSpec context in `openspec/config.yaml`

## 2. sciencekit_math module splits (behavior-preserving)

- [x] 2.1 Split `kernels/` (elementwise, reductions, scaling) into pure-dispatcher `mod.rs` + implementation files; companion `kernels_tests` at end; suite green
- [x] 2.2 Split `pairwise/` (squared-euclidean, manhattan, cosine, simd dot) into pure-dispatcher `mod.rs` + implementation files; suite green
- [x] 2.3 Split `layout/` into pure-dispatcher `mod.rs` + `memory_layout.rs`; suite green
- [x] 2.4 Split `sparse_ops/` into pure-dispatcher `mod.rs` + implementation files; suite green

## 3. sciencekit_common module splits (behavior-preserving)

- [x] 3.1 Split `errors/` into pure-dispatcher `mod.rs` + `error_kind.rs`; suite green
- [x] 3.2 Split `execution/` (modes, context, resolver) into pure-dispatcher `mod.rs` + implementation files; suite green
- [x] 3.3 Split `fit_traits/` into pure-dispatcher `mod.rs` + implementation files; suite green
- [x] 3.4 Split `scorer_traits/` into pure-dispatcher `mod.rs` + implementation files; suite green
- [x] 3.5 Split `data_view/` into pure-dispatcher `mod.rs` + implementation files; suite green
- [x] 3.6 Split `batching/` (data batch, lazy source, mappable source) into pure-dispatcher `mod.rs` + implementation files; suite green
- [x] 3.7 Split `label_table/` into pure-dispatcher `mod.rs` + implementation files; suite green
- [x] 3.8 Split `target_view/` into pure-dispatcher `mod.rs` + implementation files; suite green
- [x] 3.9 Split `sk_float/` into pure-dispatcher `mod.rs` + `definition.rs`; suite green

## 4. Verification

- [x] 4.1 Confirm every `mod.rs` in the two crates is now a pure dispatcher (no implementation inline)
- [x] 4.2 Confirm any inline `#[cfg(test)]` block sits at the end of its file
- [x] 4.3 Run the full workspace test suite; all tests green and unchanged in behavior
- [x] 4.4 Confirm `backend/` was not touched (owned by `backend-kernel-expansion`)