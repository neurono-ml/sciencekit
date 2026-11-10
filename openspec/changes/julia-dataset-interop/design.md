# Design

## Context

See `proposal.md` — Why. Rust and Julia can meet in two ways: embed Julia (`jlrs`) or call a Rust
cdylib from Julia through a C ABI. Binding users want a normal Julia package, so this design uses a
**C ABI cdylib** called from `ScienceKit.jl`; Julia owns the process and Rust is a library.

Julia tables implement Tables.jl (`Tables.istable`, `Tables.columns`, `Tables.columnnames`,
`Tables.getcolumn`). Julia matrices are column-major. `SparseMatrixCSC` stores column pointers,
row indices, and values. `CategoricalArray` stores levels plus integer references; `missing` is a
value of the element type (`Union{T, Missing}`).

The Arrow C Data Interface (`ArrowArrayStream`) is the shared zero-copy ABI; `Arrow.jl` support for
it is still maturing, so the design includes an Arrow IPC fallback.

## Goals / Non-Goals

**Goals**

- One Tables.jl path, no per-package adapters.
- Layout-honest conversions: report transpose/copy rather than pretending.
- Categorical and missing semantics shared with Rust and Python.
- Bounded memory for large Arrow tables.

**Non-Goals**

- No Julia package for every algorithm in this change; it adds the dataset surface to the binding.
- No automatic conversion of nested columns or arbitrary Julia structs.
- No Julia-side model training; Rust owns computation.
- No `jlrs` embedding; the C ABI keeps the dependency surface small.
- No zero-copy for column-major → row-major without a transpose, which is a copy.

## Decisions

### C ABI over Arrow C Data Interface

The cdylib exposes `sk_julia_*` functions that accept an `ArrowArrayStream` plus a selection
descriptor and return an `ArrowArrayStream` of results. `ScienceKit.jl` builds the input stream from
Tables.jl data using `Arrow.jl` or a small Julia-side Arrow writer when the C interface is
unavailable. Strings and error messages cross as C strings.

Alternative: `jlrs` embedding from Rust. Rejected because it inverts ownership and makes packaging
harder for Julia users.

### Tables.jl → Arrow columns

`Tables.columns(table)` yields column objects. For each selected column, the Julia side builds an
Arrow array: numeric columns directly, `Union{T,Missing}` with a validity bitmap, strings as
`String`/dictionary, categoricals as dictionary with levels. This keeps the Rust path identical to
the Arrow adapter.

For matrices, Julia's column-major layout maps naturally to Arrow columns (each column is
contiguous), so a `Matrix{Float64}` can be exported as `n` Arrow columns with zero copy per column.
The Rust side then materializes row-major per batch, which is the documented copy for the dense
view. For inference with a single feature, a column can be borrowed directly.

### Sparse conversion

`SparseMatrixCSC` → Arrow struct columns (`indptr`, `indices`, `data`, shape) or a direct C ABI
form. The Rust side builds `sprs::CsMat` in CSR by transposing: CSC of matrix A equals CSR of Aᵀ, so
the adapter either transposes the logical matrix or builds CSR with row pointers derived from CSC
column pointers after a transpose. The design states: the binding accepts CSC and converts to CSR
explicitly, preserving stored zeros; for a transposed interpretation, rows and columns swap and the
caller is told.

### Missing and categorical

`missing` → validity bitmap → neutral policy. `CategoricalArray` levels → Arrow dictionary →
label table. The Julia binding returns predicted categories by indexing the levels, matching the
Python behavior.

### Result return

Rust returns an `ArrowArrayStream`; Julia imports it with `Arrow.Table` and converts to `Matrix`,
`Vector`, `DataFrame` (`DataFrame(arrow_table)`), or keeps the Arrow table. Sparse results return
as CSC reconstructed from CSR (transpose again).

### Threading

The cdylib functions are stateless; each call owns its Rust state. Julia threads calling
concurrently use independent handles. Rust's rayon pool is per-process; nested threading is
bounded by the execution plan.

## Risks / Trade-offs

- **Arrow.jl C Data Interface maturity** → Mitigation: version-gated feature detection and an IPC
  fallback; tests run both paths when available.
- **Column-major → row-major copy** → Mitigation: document and report; per-batch copies are
  bounded by the streaming block size.
- **CSC↔CSR transpose bugs** → Mitigation: property tests comparing against dense multiplication
  and shape checks.
- **C ABI error handling** → Mitigation: every function returns a status code and a message pointer;
  no Rust panic crosses the boundary (`catch_unwind`).
- **Julia GC moving data** → Mitigation: Julia arrays are not moved once allocated; the binding
  keeps references alive for the call and copies anything that must outlive it.

## Migration Plan

New package and cdylib; additive. Documentation states supported Julia and Arrow.jl versions.

## Open Questions

- Whether to ship prebuilt cdylibs through Julia's artifact system in the first release. Deferred;
  source builds with `cargo` are the initial path.
