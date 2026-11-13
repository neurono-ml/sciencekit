## Why

Python is the primary community of scikit-learn users, and its datasets live in numpy arrays, pandas
DataFrames, SciPy sparse matrices, pyarrow tables, Polars frames, Dask collections, and PySpark
DataFrames. A Python binding that accepts only `ndarray` would force every user to convert and
would break the zero-copy promise on the most common paths.

This change defines the dataset compatibility surface of `sciencekit_python`: convert each Python
table representation into the neutral tabular seam, and return results in the representation the
user expects.

## What Changes

- **Python dataset adapters** in `sciencekit_python` (PyO3), built on the Arrow C Data Interface
  exported by `arrow-interop`.
- **numpy**: `ndarray` of `float32`/`float64` accepted via `PyReadonlyArray2` with a zero-copy
  borrow when C-contiguous; non-contiguous or wrong-dtype arrays produce an explicit
  copy-or-reject decision, never a silent copy.
- **pandas**: `DataFrame`/`Series` accepted through the `__arrow_c_stream__` PyCapsule protocol
  (pandas ≥ 2.2 / 3.0), with `to_numpy`/dtype-based fallback for older pandas; nullable dtypes map
  to the missing-value policy; categorical dtype maps to the label table; the index is ignored
  unless explicitly named as a feature.
- **SciPy sparse**: `csr_matrix`/`csc_matrix`/`coo_matrix` accepted by reading `data`, `indices`,
  and `indptr` buffers; CSR maps zero-copy to `SKDataView::Sparse`, CSC/COO convert explicitly.
- **pyarrow**: `Table`, `RecordBatch`, `RecordBatchReader`, and any object implementing
  `__arrow_c_stream__` / `__arrow_c_array__` accepted zero-copy; this also covers Polars (Python),
  DuckDB relations, and other PyCapsule producers.
- **Polars (Python)**: `polars.DataFrame`/`LazyFrame` accepted through the same PyCapsule protocol;
  lazy frames collect in batches on a reader thread.
- **Dask**: `dask.dataframe.DataFrame` accepted by iterating partitions and feeding the streaming
  source; each partition converts to Arrow/pandas.
- **PySpark**: `pyspark.sql.DataFrame` accepted via Spark 4 `toArrow()` or the
  `_collect_as_arrow()` path; for out-of-core use, `mapInArrow` feeds batches; driver memory is
  bounded and the distributed path is documented.
- **Return values**: predictions/transforms return numpy arrays by default, with `as_frame`,
  `as_arrow`, `as_polars`, and `as_sparse` options; Arrow-backed results implement
  `__arrow_c_stream__`.
- **GIL**: conversion happens while holding the GIL; fitting/transforming releases it with
  `py.allow_threads`, and data is owned before release.
- **Errors**: `SKError` variants map to Python exceptions (`ValueError`, `TypeError`,
  `KeyError` for unknown column) with the column name in the message.
- **Tests**: conversion parity for every representation, zero-copy assertions, null/categorical
  behavior, large and small data, threading, `f32`/`f64`.

## Capabilities

### New Capabilities

- `python-dataset-interop`: Python tabular and array representations as sciencekit inputs and
  outputs, including zero-copy paths, missing-value policy, lazy/distributed sources, and Python
  result types.

### Modified Capabilities

<!-- none -->

## Impact

- **Dependencies:** `pyo3`, `numpy` (PyO3 bindings), optional `pyarrow`/`pandas`/`polars`/`dask`/
  `pyspark` extras; the Rust side depends on `sciencekit_interop` with `arrow`.
- **New code:** `crates/sciencekit_python/src/datasets/` (`mod`, `numpy`, `pandas`, `scipy_sparse`,
  `arrow`, `polars`, `dask`, `pyspark`, `results`, companion `*_tests.rs`) and Python stubs.
- **Sequencing:** depends on `interop-foundation`, `arrow-interop`, and the existing
  `sciencekit_python` crate skeleton (per-algorithm bindings). If the skeleton is absent, this
  change creates it as a minimal crate.
- **Acceptance (PRD §8.7/§10.3):** every representation fits the same estimator with equal results;
  numpy and Arrow paths verified zero-copy; large and small data; concurrent calls; `f32`/`f64`;
  model export and metrics unaffected.
