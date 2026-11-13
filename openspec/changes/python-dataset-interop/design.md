# Design

## Context

See `proposal.md` — Why. The Rust side provides `sciencekit_interop` (`dataset-interop-core`) and
the Arrow C Data Interface (`arrow-interop`). The binding is PyO3-based (`sciencekit_python`), which
may not exist yet; per-algorithm bindings are separate changes, so this change either extends the
crate or creates a minimal skeleton if absent.

Python-side facts that shape the design:

- `numpy` exposes array buffers; PyO3's `numpy` crate provides `PyReadonlyArray2<F>` and
  `as_array()` for a zero-copy borrow, valid only while the GIL is held and the array is not
  resized.
- pandas ≥ 2.2 and Polars ≥ 1.3 implement the Arrow PyCapsule interface
  (`__arrow_c_stream__`), as do pyarrow tables/readers and DuckDB relations.
- SciPy CSR stores `data`, `indices`, `indptr` as numpy arrays; `csc_matrix` and `coo_matrix`
  require conversion.
- Dask DataFrames expose partitions; PySpark 4 exposes `toArrow()` and `_collect_as_arrow()`; Spark
  Connect exposes `toArrow()`.

## Goals / Non-Goals

**Goals**

- One conversion dispatcher that recognizes all supported Python representations.
- Zero-copy for numpy contiguous arrays and Arrow PyCapsule producers.
- Bounded memory for lazy/distributed sources.
- Results return as numpy by default, with explicit format options.
- GIL released for all heavy computation.

**Non-Goals**

- No automatic index preservation; the pandas index is metadata and is ignored unless named.
- No conversion of Python lists, dicts of columns, or arbitrary objects in this change.
- No distributed scheduling of sciencekit across a Spark/Dask cluster; only data consumption.
- No pandas < 2.0 guarantee; fallback is best-effort and documented.
- No nested columns as features.

## Decisions

### Dispatch order: Arrow PyCapsule → numpy → SciPy sparse → specialized

The dispatcher first checks `__arrow_c_stream__`/`__arrow_c_array__`, which covers pandas ≥ 2.2,
Polars, pyarrow, DuckDB, and any future producer with no code changes. Then numpy, then SciPy
sparse, then Dask/PySpark, then an error listing supported types. This ordering avoids pandas/polars
version-specific code and maximizes zero-copy.

Alternative: import and `isinstance`-check pandas/polars first. Rejected because it hard-codes every
library and breaks on new producers.

### Arrow C stream consumption

The binding calls the object's `__arrow_c_stream__`, gets a PyCapsule, unwraps it into an
`FFI_ArrowArrayStream` using the capsule name `arrow_array_stream`, and imports it with arrow-rs.
The imported stream is owned by Rust; the capsule is released when consumed per the PyCapsule
protocol. Batches convert through `SKTabularBatch`.

### numpy borrow lifetime

`PyReadonlyArray2<F>` keeps the numpy object alive through the borrow and the binding increments its
reference count. For a contiguous array, the binding builds `SKDataView::Dense` borrowing the numpy
buffer and releases the GIL for the estimator call. This is sound because the buffer is owned by the
numpy object, cannot be resized while the reference is held, and the estimator only reads it. The
documented contract is that the caller must not mutate the array during the call; the read-only
borrow enforces this in Rust. When the estimator needs owned streaming data, the binding copies
into an owned batch under the GIL and then releases it; that path is reported as a copy.

### Missing values and categoricals

pandas nullable dtypes arrive as Arrow validity bitmaps through the PyCapsule path, so the neutral
policy applies uniformly. For the `to_numpy` fallback, `NaN`/`pd.NA` detection happens in Rust; the
adapter rejects object columns that are not strings.

Categorical pandas dtype (`CategoricalDtype`) maps to Arrow dictionary, so the label table is the
category order. Predictions map back with `numpy.take(categories, indices)`.

### SciPy sparse

`csr_matrix` exposes `data`, `indices`, `indptr`, `shape` as attributes; reading them yields numpy
arrays. The binding builds `sprs::CsMatView` over the borrowed slices; `indices`/`indptr` are
`int32` or `int64` depending on SciPy's index dtype, so the adapter accepts both and converts index
arrays to `usize` only if needed (CSR view requires `usize` in sprs). If conversion is needed, it is
reported. `csc_matrix`/`coo_matrix` convert through SciPy's `.tocsr()` in Python.

### Dask and PySpark

Dask: `df.to_delayed()` / `df.partitions` iterated in Python, each partition converted to Arrow and
fed to a `SKTabularSource`. PySpark: prefer `toArrow()` (Spark 4) for driver-sized data; for
out-of-core, expose `mapInArrow` integration where the user's pipeline calls a sciencekit UDF per
Arrow batch; the binding also accepts a `RecordBatchReader` produced by `_collect_as_arrow()` when
available. Tests use local Spark (`local[2]`) and skip when pyspark is not installed.

### Errors and result types

`SKError::UnknownColumn` → `KeyError`; `UnsupportedColumnType`/`NullValueInColumn` → `TypeError`;
`ShapeMismatch`/`UnevenColumnLengths` → `ValueError`. Result formatting is a small enum
(`Numpy`, `Pandas`, `Arrow`, `Polars`, `Sparse`) implemented in one module.

## Risks / Trade-offs

- **GIL and zero-copy lifetime** → Mitigation: document immutability during the call; keep the
  numpy reference alive; tests mutate-detection is out of scope.
- **pandas version differences** → Mitigation: PyCapsule first; fallback tested on the oldest
  supported pandas; explicit error when a dtype cannot be represented.
- **PySpark driver memory** → Mitigation: document `toArrow()` as driver-collect; provide and test
  the `mapInArrow` path; never claim out-of-core for the driver path.
- **Optional dependency matrix** → Mitigation: install extras per test job; conversion modules are
  feature-gated at the Python import level, not the Rust crate.
- **Arrow capsule ownership bugs** → Mitigation: one small wrapper, release-once tests, and reuse
  of the arrow-rs FFI tests.

## Migration Plan

Additive. Existing numpy-only binding paths keep working; the dispatcher is a superset. Python
stubs and documentation describe the supported types and extras.

## Open Questions

- Whether to expose an explicit `copy=True` argument globally or per representation. Deferred:
  per-representation options (`contiguous=True`, `cast=True`).
