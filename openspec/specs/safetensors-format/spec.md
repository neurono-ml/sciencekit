# safetensors-format Specification

## Purpose
Defines the extended Safetensors model format: single-file encoding, metadata, sharding,
padding, compression and import back into the model state.

## Requirements

### Requirement: Single-file round-trip

The library SHALL write a model state to a Safetensors file and read it back into an
equivalent state, preserving every tensor name, dtype, shape and value.

#### Scenario: Round-trip fidelity

- **WHEN** a state is written and read back
- **THEN** its metadata, tensor names, dtypes, shapes and values are identical

#### Scenario: Dtype is preserved

- **WHEN** an `F32` state is written and read as `F32`
- **THEN** every tensor retains its `F32` dtype

### Requirement: Metadata header

The file header SHALL contain the estimator name, format version, scalar dtype, feature
count, class/label tables and typed hyperparameters in addition to the Safetensors tensor
table.

#### Scenario: Metadata survives

- **WHEN** a classifier state is written and read
- **THEN** its class table and hyperparameters are present in the read metadata

#### Scenario: Unknown header field is ignored

- **WHEN** a reader encounters an unknown metadata field
- **THEN** it ignores it and preserves the known fields

### Requirement: Sharding

The writer SHALL split a large state into immutable shards when the configured size
threshold is exceeded and SHALL write an index that maps tensor names to shards; the reader
SHALL merge the shards.

#### Scenario: Large state is sharded

- **WHEN** a state exceeds the shard threshold
- **THEN** multiple shard files and an index are produced

#### Scenario: Reader merges shards

- **WHEN** a shard set is read
- **THEN** the resulting state equals the original single-file state

### Requirement: Header padding

The writer SHALL reserve padding so small tensor additions can be written in place while
the padding lasts; when it does not fit, a new shard SHALL be created.

#### Scenario: Append fits in padding

- **WHEN** a small tensor is added to a padded file with enough reserved space
- **THEN** it is written in place without rewriting the payload

#### Scenario: Append overflows padding

- **WHEN** the addition does not fit
- **THEN** the writer creates a new shard and updates the index

### Requirement: Compression

The library SHALL support gzip, Brotli and Snappy compressed variants behind optional
features, and SHALL read a compressed file transparently.

#### Scenario: Compressed round-trip

- **WHEN** a state is written with a compression feature and read back
- **THEN** the state is identical to the uncompressed round-trip

#### Scenario: Missing feature is explicit

- **WHEN** a compressed file is read without the matching feature
- **THEN** the reader returns `UnsupportedOperation` with guidance

### Requirement: Validation

The reader SHALL reject truncated files, inconsistent offsets, duplicate tensor names,
dtype/shape mismatches, missing shards and corrupt compression payloads with structured
errors.

#### Scenario: Truncated file

- **WHEN** the file ends before the declared tensor payload
- **THEN** the reader returns a structured error

#### Scenario: Missing shard

- **WHEN** the index references a shard that does not exist
- **THEN** the reader returns `EmptyInput` or a structured I/O error

### Requirement: Concurrency and compatibility

Reads SHALL be safe under concurrent calls, and the format SHALL expose a version so a
reader can reject an incompatible future version explicitly.

#### Scenario: Concurrent reads agree

- **WHEN** several threads read the same file
- **THEN** every state equals the single-threaded read

#### Scenario: Future version is rejected

- **WHEN** a file declares a newer major format version
- **THEN** the reader returns `UnsupportedRepresentation` with guidance
