## 1. Feature, dtype mapping, eager import

- [ ] 1.1 Add the `polars` feature (with `lazy`, `streaming`, `ndarray`, `async`, dtype coverage) and the CI feature-matrix entry; document the supported Polars range.
- [ ] 1.2 TDD dtype mapping: `Float32`/`Float64`/`Int64`/`Boolean`/`String`/`Categorical`/`Enum`; lossless integer widening; structured rejection of date/list/struct.
- [ ] 1.3 TDD eager `DataFrame` import via the Arrow adapter: names, order, rows, values.
- [ ] 1.4 TDD eager `Series` import: single-column batch, series name preserved, null policy applied.
- [ ] 1.5 TDD single-chunk zero-copy: `to_ndarray` path aliases the buffer; assert pointer identity; multi-chunk materializes and reports a copy.
- [ ] 1.6 TDD categorical and string columns: codes + label table, deterministic round-trip.
- [ ] 1.7 TDD null policy: reject, NaN, categorical sentinel, constant fill.

## 2. Lazy streaming source

- [ ] 2.1 TDD lazy import through the async bridge: ordered batches, configured chunk size, final marking.
- [ ] 2.2 TDD `maintain_order = true` behavior under a parallel lazy plan.
- [ ] 2.3 TDD reopenable lazy source: two passes produce identical batches.
- [ ] 2.4 TDD projection pushdown: only selected columns appear in the executed plan; selection order preserved.
- [ ] 2.5 TDD lazy-versus-eager equivalence on the same frame for a representative estimator.
- [ ] 2.6 TDD cancellation: dropping the lazy source stops collection and terminates the reader thread.

## 3. Export and validation

- [ ] 3.1 TDD export dense predictions and transformed blocks to `Series`/`DataFrame`.
- [ ] 3.2 TDD export CSR blocks in the portable triple form and round-trip through import.
- [ ] 3.3 TDD export streaming batches and supervised batches.
- [ ] 3.4 TDD large and small frames, `f32`/`f64`, concurrent readers.
- [ ] 3.5 Run the CI gates: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`, `cargo test --workspace`.
- [ ] 3.6 Add the Polars interop book chapter and the streaming-collection `Development Explained` chapter; register in `SUMMARY.md`.
- [ ] 3.7 Update `CHANGELOG.md` and open the ADR issue for this change.
