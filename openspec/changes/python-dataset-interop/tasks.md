## 1. Dispatcher and numpy

- [x] 1.1 Create or extend `crates/sciencekit_python` with a `datasets` module and a representation dispatcher (Arrow PyCapsule → numpy → SciPy sparse → Dask/PySpark → error).
- [x] 1.2 TDD numpy contiguous `float32`/`float64` borrow: values, zero-copy pointer identity, immutable-during-call contract.
- [x] 1.3 TDD numpy non-contiguous and wrong-dtype: explicit error with suggested contiguous copy/cast; opt-in copy path.
- [x] 1.4 TDD dispatcher error listing supported representations.

## 2. pandas and Arrow PyCapsule

- [x] 2.1 TDD `pyarrow.Table`/`RecordBatch`/`RecordBatchReader` import through the Arrow C stream; batch-by-batch consumption.
- [x] 2.2 TDD pandas DataFrame/Series through `__arrow_c_stream__`: columns, values, index ignored, feature/target roles.
- [x] 2.3 TDD pandas nullable dtypes: `pd.NA` under reject/NaN/fill policies.
- [x] 2.4 TDD pandas categorical target: label table, predictions mapped back to categories.
- [x] 2.5 TDD older-pandas fallback path (`to_numpy`) and its documented copy behavior.
- [ ] 2.6 TDD generic PyCapsule producer (a minimal class implementing `__arrow_c_stream__`) is accepted.

## 3. SciPy sparse

- [ ] 3.1 TDD `csr_matrix` import: `data`/`indices`/`indptr` borrowed, sparse view, stored zeros preserved.
- [ ] 3.2 TDD `int32` and `int64` index dtypes; conversion reporting.
- [ ] 3.3 TDD `csc_matrix` and `coo_matrix` explicit conversion to CSR.
- [ ] 3.4 TDD sparse fit/transform round-trip and result returned as `csr_matrix` when requested.

## 4. Polars, Dask, PySpark

- [ ] 4.1 TDD Polars Python `DataFrame` through the PyCapsule interface; zero intermediate pandas.
- [ ] 4.2 TDD Polars `LazyFrame` collected in batches on a reader thread; chunk size and final marking.
- [ ] 4.3 TDD Dask DataFrame: partition order, bounded resident memory, streamed fit equals concatenated fit.
- [ ] 4.4 TDD PySpark small frame via `toArrow()`; parity with pandas.
- [ ] 4.5 TDD PySpark distributed path via `mapInArrow`; skip when pyspark is absent; document driver-memory limits.

## 5. Results, threading, errors, validation

- [ ] 5.1 TDD result formats: numpy default, pandas, pyarrow, Polars, SciPy sparse; values round-trip.
- [ ] 5.2 TDD Arrow-backed results implement `__arrow_c_stream__` and are consumable by pyarrow/polars.
- [ ] 5.3 TDD GIL release: two Python threads fit concurrently; both succeed; no deadlock.
- [ ] 5.4 TDD error mapping: unknown column → `KeyError`, unsupported dtype → `TypeError`, shape mismatch → `ValueError`.
- [ ] 5.5 TDD `float32`/`float64` and large/small data across representations.
- [ ] 5.6 Run the CI gates plus the Python test matrix with optional extras.
- [ ] 5.7 Add the Python interop book chapter and the `Development Explained` chapter on binding representations; register in `SUMMARY.md`; add Python stubs and docs.
- [ ] 5.8 Update `CHANGELOG.md` and open the ADR issue for this change.
