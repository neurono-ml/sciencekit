## Why

PRD §4.4 requires native out-of-core access with random reads (`SKMappableSource`,
`OutOfCoreMemoryMapped`), and the streaming spec deliberately leaves the concrete memory-mapped
implementation to the interop layer. Algorithms such as k-means, k-NN, DBSCAN, and decision trees
need rows by index without scanning the file, and datasets larger than RAM must be usable without a
copy.

This change supplies the memory-mapped adapters: raw binary, NumPy `.npy`, and random-access Arrow
IPC, all exposing zero-copy row views.

## What Changes

- **Memmap feature on `sciencekit_interop`** (`memmap`) using `memmap2`, plus `npyz` for `.npy`.
- **Raw binary source**: memory-maps a file of fixed-width scalars with a sidecar shape/dtype
  descriptor (or explicit parameters) and implements `SKMappableSource<F>` with O(1) row access.
- **NumPy `.npy` source**: memory-maps C-order `float32`/`float64`/`int64` arrays for both
  `SKMappableSource` and sequential `SKLazySource` (row-block batches).
- **Arrow IPC random access**: memory-maps an Arrow IPC file, indexes record batches once, and
  serves rows by index; sequential iteration reuses the same mapping.
- **Sequential memory-mapped source**: any of the above can also feed `SKLazySource` with a
  configurable block size, so out-of-core fits use the streaming driver.
- **Validation**: file length versus declared shape, dtype match, endianness (little-endian only
  for raw), magic/version for `.npy`, truncated file errors.
- **Tests**: O(1) random access, concurrent readers, small and large files, truncated and mismatched
  files, `f32`/`f64`, and sequential/random equivalence.

## Capabilities

### New Capabilities

- `memory-mapped-sources`: memory-mapped raw binary, NumPy `.npy`, and Arrow IPC files as
  zero-copy random-access and sequential sciencekit sources, with validation and structured errors.

### Modified Capabilities

<!-- none; SKMappableSource already exists in streaming-batches -->

## Impact

- **Dependencies:** `memmap2`, `npyz` behind the `memmap` feature; Arrow IPC behind `arrow`.
- **New code:** `crates/sciencekit_interop/src/memmap/` (`mod`, `raw_binary`, `numpy`, `arrow_ipc`,
  `validation`, companion `*_tests.rs`).
- **Public API:** added behind the feature.
- **Sequencing:** depends on `interop-foundation`; the Arrow IPC part depends on `arrow-interop`.
- **Acceptance (PRD §8.7/§10.3):** files larger than RAM stream in bounded batches; random access
  does not read preceding rows; concurrent reads are safe; `f32`/`f64`; model export and metrics
  unaffected.
