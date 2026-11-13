## 1. JNI ABI and JVM modules

- [ ] 1.1 Create `crates/sciencekit_jvm_abi` (JNI cdylib): Arrow IPC buffer input/output, status code, UTF-8 error message, `catch_unwind`.
- [ ] 1.2 Create the `java/` multi-module build (`sciencekit-java`, `sciencekit-scala`) with JNI loader and native library packaging.
- [ ] 1.3 TDD error propagation: Rust errors become JVM exceptions; a forced panic becomes `IllegalStateException` without crashing the JVM.
- [ ] 1.4 TDD native memory: handles released exactly once on success and failure; repeated calls do not leak.

## 2. Arrow transport and Spark

- [ ] 2.1 TDD Arrow IPC round-trip: a Java-built Arrow table crosses to Rust and back with identical schema and values.
- [ ] 2.2 TDD Spark 4 driver path via `toArrow()`: names, values, selection; parity with local Java data.
- [ ] 2.3 TDD Spark partition path via `mapInArrow`: multiple partitions, bounded driver memory, ordered results.
- [ ] 2.4 TDD Spark unsupported SQL type rejection with a clear message.
- [ ] 2.5 TDD Spark result creation from returned Arrow batches.

## 3. Smile, Breeze, ND4J

- [ ] 3.1 TDD Smile `DataFrame`: numeric, boolean, string, and categorical columns; selection and rejection paths.
- [ ] 3.2 TDD Breeze `DenseMatrix`/`DenseVector`: values, shape, copy reporting.
- [ ] 3.3 TDD Breeze `CSRMatrix`/`CSCMatrix`: conversion to CSR, stored zeros preserved.
- [ ] 3.4 TDD ND4J row-major `FLOAT`/`DOUBLE` direct-buffer path; zero per-element copies reported.
- [ ] 3.5 TDD ND4J non-contiguous/column-major conversion and reporting.
- [ ] 3.6 TDD result formats: ND4J, Breeze, Smile, Spark, Arrow.

## 4. Threading, validation, documentation

- [ ] 4.1 TDD concurrent calls from multiple JVM threads on independent data.
- [ ] 4.2 TDD `float`/`double` across all representations; cast reporting.
- [ ] 4.3 TDD large and small data; Spark local mode with several partitions.
- [ ] 4.4 Run Rust CI gates plus the JVM test suite (`sbt`/Maven) with provided dependencies.
- [ ] 4.5 Add the JVM interop book chapter and the `Development Explained` chapter on JNI and Arrow transport; register in `SUMMARY.md`; add Java/Scala API docs.
- [ ] 4.6 Update the PRD binding roadmap and `CHANGELOG.md`; open the ADR issue for this change.
