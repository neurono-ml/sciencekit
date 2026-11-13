# Spec Delta

## Purpose

Defines memory-mapped datasets — raw binary, NumPy `.npy`, and Arrow IPC — as zero-copy
random-access and sequential sources for sciencekit, with explicit layout validation and structured
errors when a file does not match its declared shape or type.

## ADDED Requirements

### Requirement: Raw binary matrix source with O(1) row access

The adapter SHALL memory-map a fixed-width scalar file and implement the library's random-access
source contract, returning a borrowed row view in constant time without reading preceding rows.

#### Scenario: Row access does not scan
- **WHEN** row 900 000 of a 1 000 000-row file is requested
- **THEN** the access reads only that row's bytes and returns a view into the mapping

#### Scenario: Shape mismatch is rejected
- **WHEN** the declared shape requires more bytes than the file contains
- **THEN** opening fails with a structured error naming the file and the expected byte length

### Requirement: NumPy `.npy` arrays are read without copying

The adapter SHALL open C-order `.npy` files of `float32`, `float64`, and `int64`, validate the
header, and expose the array as a memory-mapped source; float dtypes matching the model scalar SHALL
be viewed without copying.

#### Scenario: float64 npy maps zero-copy
- **WHEN** an `f64` `.npy` matrix is opened for an `f64` model
- **THEN** the returned rows alias the mapped file

#### Scenario: Fortran-order npy is rejected
- **WHEN** a `.npy` file declares Fortran order
- **THEN** opening fails with a structured error suggesting a C-order conversion

#### Scenario: Non-float npy is opted in explicitly
- **WHEN** an `int64` `.npy` file is opened for an `f64` model without explicit cast
- **THEN** opening fails with an unsupported-type error and the cast option is the documented path

### Requirement: Arrow IPC files provide random row access

The adapter SHALL memory-map an Arrow IPC file, index its record batches once, and serve a row by
index by locating the batch that contains it; the first access SHALL NOT scan all batches.

#### Scenario: Indexed batch lookup is direct
- **WHEN** a row in the last batch is requested
- **THEN** the adapter computes the batch from the cumulative row offsets and reads only that batch

#### Scenario: Truncated IPC file fails on open or read
- **WHEN** the mapped file ends inside a record batch
- **THEN** opening or the affected read returns a structured error, never a panic

### Requirement: Mapped sources also expose sequential blocks

The adapter SHALL expose every mapped source as a sequential fallible source delivering owned
batches of a configured maximum row count, so streaming estimators work without a second reader.

#### Scenario: Sequential and random access agree
- **WHEN** the same file is read sequentially and row by row
- **THEN** the concatenated sequential rows equal the random-access rows in order

#### Scenario: Block size is respected
- **WHEN** a 10 000-row mapped file is streamed with a block size of 512
- **THEN** each batch has at most 512 rows and the last batch is marked final

### Requirement: Mappings are safe under concurrency

Mapped sources SHALL be `Send` and `Sync` where the underlying mapping allows it, and concurrent row
reads SHALL return identical, valid data without data races.

#### Scenario: Concurrent readers
- **WHEN** eight threads read disjoint rows from the same source
- **THEN** every returned row matches the expected values and no race is reported

### Requirement: Validation errors are structured

Every opening failure SHALL use the central taxonomy with distinct variants for missing file,
permission failure, unsupported dtype, non-little-endian raw data, shape/length mismatch, and
malformed header, each naming the file.

#### Scenario: Wrong endianness is rejected
- **WHEN** a raw file is declared big-endian
- **THEN** opening fails with a structured error naming the file and the supported endianness

#### Scenario: Malformed npy header
- **WHEN** a `.npy` file has a corrupt header
- **THEN** opening fails with a malformed-header error, not a panic
