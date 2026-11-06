# Spec Delta

## Purpose

Defines the tabular source contract and the feature-gated Polars, DataFusion, Arrow and
Parquet conversions, including column kinds, missing values and categorical mapping.

## ADDED Requirements

### Requirement: Tabular source contract

The library SHALL expose a tabular source with a schema of named columns and kinds
(numeric, categorical, boolean, ignored), a numeric feature block and an optional
categorical view.

#### Scenario: Numeric columns become a feature block

- **WHEN** a table with numeric columns is converted
- **THEN** the numeric columns form a dense feature block in schema order

#### Scenario: Categorical columns become a categorical view

- **WHEN** a table has string/categorical columns
- **THEN** those columns form a categorical view with a documented missing sentinel

#### Scenario: Ignored columns are excluded

- **WHEN** a column is marked ignored
- **THEN** it is absent from the feature and categorical outputs

### Requirement: Polars conversions

The library SHALL convert Polars `DataFrame` and `LazyFrame` into the tabular source
behind the `polars` feature, with column selection and missing-value mapping.

#### Scenario: DataFrame conversion

- **WHEN** a Polars DataFrame with numeric and categorical columns is converted
- **THEN** the resulting schema, feature block and categorical view match the DataFrame

#### Scenario: LazyFrame batches

- **WHEN** a LazyFrame is converted in streaming mode
- **THEN** batches are produced through the source contract without collecting the whole
  frame

### Requirement: DataFusion and Arrow conversions

The library SHALL convert DataFusion `DataFrame`/`RecordBatch` and Arrow/Parquet data
behind their features, preserving schema and row order.

#### Scenario: RecordBatch conversion

- **WHEN** an Arrow RecordBatch is converted
- **THEN** its numeric and categorical columns map to the same schema/feature outputs

#### Scenario: Parquet through DataFusion

- **WHEN** a Parquet file is read through DataFusion
- **THEN** its columns are converted with the same rules as a DataFrame

### Requirement: Missing values

Numeric missing values SHALL map to NaN; categorical missing values SHALL map to the
documented missing sentinel; nulls in unsupported columns SHALL be rejected.

#### Scenario: Numeric null

- **WHEN** a numeric column contains a null
- **THEN** the feature block contains NaN at that position

#### Scenario: Categorical null

- **WHEN** a categorical column contains a null
- **THEN** the categorical view uses the missing sentinel

### Requirement: Validation and concurrency

Conversions SHALL reject empty tables, missing selected columns, unsupported dtypes,
schema/row mismatches and non-finite values outside the missing policy; converted sources
SHALL be safely shareable.

#### Scenario: Unsupported dtype

- **WHEN** a selected column has an unsupported dtype
- **THEN** conversion returns `UnsupportedRepresentation` naming the column

#### Scenario: Concurrent reads agree

- **WHEN** several threads read the same materialized source
- **THEN** every read equals the single-threaded read
