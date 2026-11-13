# Spec Delta

## Purpose

Defines which Python dataset representations the sciencekit binding accepts and returns, how each is
converted (zero-copy where the representation allows), how nulls and categoricals are handled, and
how lazy and distributed Python sources feed the streaming contract.

## ADDED Requirements

### Requirement: numpy arrays are accepted without copying when contiguous

The binding SHALL accept `numpy.ndarray` of `float32` and `float64` with two dimensions and C order
by borrowing the buffer, and SHALL reject a non-contiguous or wrong-dtype array unless the caller
explicitly requests a copy or cast.

#### Scenario: Contiguous float64 array is borrowed
- **WHEN** a C-contiguous `float64` array is passed to `fit`
- **THEN** no element copy occurs and the fitting sees the same values

#### Scenario: Non-contiguous array is not silently copied
- **WHEN** a transposed or strided array is passed without an explicit copy request
- **THEN** the call fails with an error suggesting a contiguous copy

### Requirement: pandas frames are accepted through the Arrow protocol

The binding SHALL accept `pandas.DataFrame` and `pandas.Series` through the Arrow PyCapsule
interface when available, falling back to a documented conversion for older pandas, and SHALL map
nullable and categorical dtypes through the declared missing-value policy and label table.

#### Scenario: DataFrame with categoricals
- **WHEN** a pandas DataFrame with a categorical target column is passed
- **THEN** the label table matches the categories and predictions map back to them

#### Scenario: Nullable integer column
- **WHEN** a pandas nullable integer column contains `pd.NA` under the reject policy
- **THEN** the call fails with an error naming the column

#### Scenario: Index is not a feature
- **WHEN** a DataFrame with a non-default index is passed without naming the index
- **THEN** the index is ignored and only columns become features

### Requirement: SciPy sparse matrices map to sparse views

The binding SHALL accept `scipy.sparse` CSR matrices by borrowing `data`, `indices`, and `indptr`,
and SHALL convert COO and CSC explicitly, preserving stored zeros according to the declared policy.

#### Scenario: CSR is borrowed
- **WHEN** a `csr_matrix` of `float64` is passed
- **THEN** the sparse view aliases its three buffers

#### Scenario: CSC is converted explicitly
- **WHEN** a `csc_matrix` is passed
- **THEN** the binding converts it to CSR and reports the conversion, preserving values

### Requirement: Any Arrow PyCapsule producer is accepted

The binding SHALL accept `pyarrow.Table`, `pyarrow.RecordBatch`, `pyarrow.RecordBatchReader`, and
any object exporting `__arrow_c_stream__` or `__arrow_c_array__`, consuming the C stream zero-copy
and applying the standard column/nulls/dtype rules.

#### Scenario: Polars Python frame is accepted
- **WHEN** a `polars.DataFrame` is passed
- **THEN** the binding consumes its Arrow stream without an intermediate pandas conversion

#### Scenario: RecordBatchReader is streamed
- **WHEN** a `pyarrow.RecordBatchReader` over a large dataset is passed
- **THEN** batches are consumed incrementally and the whole table is never materialized

### Requirement: Dask collections stream partition by partition

The binding SHALL accept a `dask.dataframe.DataFrame` by iterating its partitions and feeding each
partition to the streaming source, preserving partition order and bounding resident memory.

#### Scenario: Dask fit is out-of-core
- **WHEN** a Dask DataFrame larger than the configured memory budget is fitted
- **THEN** partitions are consumed in order and resident memory stays bounded

### Requirement: PySpark DataFrames are accepted with a bounded driver path

The binding SHALL accept a `pyspark.sql.DataFrame`, using Spark 4 `toArrow()` or the Arrow collect
path when the result fits on the driver, and SHALL document and test the `mapInArrow` path for
distributed, out-of-core consumption.

#### Scenario: Small PySpark frame
- **WHEN** a small PySpark DataFrame is passed
- **THEN** it is converted through Arrow and produces the same result as the equivalent pandas input

#### Scenario: Distributed path is explicit
- **WHEN** a PySpark DataFrame larger than the driver memory is passed through the distributed path
- **THEN** batches are consumed per partition without collecting the full frame on the driver

### Requirement: Results return in the requested Python representation

The binding SHALL return predictions and transformed blocks as numpy arrays by default and SHALL
offer explicit options to return pandas, pyarrow, Polars, or SciPy sparse representations, round-
tripping values without loss.

#### Scenario: Default prediction type
- **WHEN** `predict` is called with no format option
- **THEN** the result is a `numpy.ndarray` of the model scalar

#### Scenario: Arrow result supports the PyCapsule protocol
- **WHEN** predictions are requested as an Arrow-backed result
- **THEN** other libraries can consume the result through `__arrow_c_stream__` without copying

### Requirement: The GIL is released around computation

The binding SHALL perform conversion while holding the GIL, take ownership of the data, and release
the GIL for the duration of fitting, transforming, and predicting.

#### Scenario: Concurrent Python threads
- **WHEN** two Python threads call `fit` concurrently on separate data
- **THEN** both run and produce correct independent models

### Requirement: Python errors are structured and named

Every conversion failure SHALL raise a Python exception (value, type, or key error as appropriate)
whose message names the offending column or array and the suggested conversion.

#### Scenario: Unknown column raises KeyError
- **WHEN** a feature name absent from the frame is requested
- **THEN** the binding raises `KeyError` naming the column

#### Scenario: Unsupported dtype raises TypeError
- **WHEN** a datetime column is selected as a numeric feature
- **THEN** the binding raises `TypeError` with the suggested cast

### Requirement: Both scalar precisions are supported

Every Python conversion SHALL support `float32` and `float64` models, casting explicitly when the
array dtype differs and reporting the cast.

#### Scenario: float32 input with float64 model
- **WHEN** a `float32` array is passed to an `f64` estimator
- **THEN** values are promoted and the conversion is reported as a cast, not zero-copy
