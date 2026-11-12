## 1. Feature, raw binary, validation

- [x] 1.1 Add the `memmap` feature (`memmap2`, `npyz`) and the CI feature-matrix entry.
- [x] 1.2 TDD raw binary descriptor: explicit parameters and sidecar JSON; dtype/row/column parsing; invalid descriptor rejection.
- [x] 1.3 TDD raw binary source: `number_of_rows`, O(1) `row`, values; unaligned/misaligned raw layout rejected.
- [x] 1.4 TDD length validation: file shorter or longer than declared shape fails with a structured error naming the file.
- [x] 1.5 TDD wrong endianness and unsupported dtype errors.

## 2. NumPy `.npy`

- [x] 2.1 TDD header parsing: magic, version, dtype descriptor, shape, C order; malformed header and truncated header errors.
- [x] 2.2 TDD float32/float64 zero-copy row views; pointer identity against the mapping.
- [x] 2.3 TDD Fortran-order rejection and int64-without-cast rejection; explicit cast path materializes.
- [x] 2.4 TDD alignment handling: aligned and misaligned header lengths; structured error when a zero-copy view is impossible.
- [ ] 2.5 TDD sequential source over the same `.npy` file; block size, final marking, values.

## 3. Arrow IPC random access

- [ ] 3.1 TDD open an Arrow IPC file, read only the footer, and build the batch index; assert the data batches are not decoded on open.
- [ ] 3.2 TDD row lookup by cumulative offsets: first, middle, last rows; no full scan.
- [ ] 3.3 TDD truncated IPC file: open fails or the affected read fails with a structured error.
- [ ] 3.4 TDD sequential source over the same IPC file; equivalence with random access.

## 4. Concurrency, acceptance, documentation

- [ ] 4.1 TDD concurrent readers: eight threads, disjoint rows, values match; `Send + Sync` compile assertions.
- [ ] 4.2 TDD small and large generated files; a file larger than the configured memory budget streams with bounded batches.
- [ ] 4.3 TDD `f32` and `f64` throughout; cast reporting.
- [ ] 4.4 Run the CI gates: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`, `cargo test --workspace`.
- [ ] 4.5 Add the memory-mapped sources book chapter and the `Development Explained` chapter on out-of-core random access; register in `SUMMARY.md`.
- [ ] 4.6 Update `CHANGELOG.md` and open the ADR issue for this change.
