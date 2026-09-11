# `sciencekit_math`: kernels, distances and the BLAS interface

Every machine-learning algorithm, from linear regression to clustering to a support-vector
machine, is at its heart just arithmetic done very, very fast over big tables of numbers.
Before sciencekit could train its first model, it needed a solid numeric engine — a crate
that knows how to walk every element of a table, how to measure how far apart two rows are,
how to multiply a sparse matrix by a dense one, and how to outsource the heaviest algebra to
a tuned library. That engine is `sciencekit_math`. This chapter opens it up from the very
beginning: first the Rust ideas it leans on, then the math, then the real objects and the
real tests that prove they work.

---

## 1. The problem we were solving

Imagine you are teaching a machine to tell apart two types of flowers by measuring their
petal lengths and widths. You store the measurements as a table: each **row** is one
flower, each **column** is one measurement (a "feature"). To build most models you must do
three kinds of work on such a table:

1. **Transform every number.** Normalize a column so every value sits between 0 and 1, or
   scale every value by some constant.
2. **Summarize.** Add up a column, or the whole table.
3. **Measure similarity.** "How far apart are these two flowers?" — the heart of clustering
   and nearest-neighbor methods.

That sounds simple, but doing it *well* is a performance minefield. A realistic dataset has
millions of rows. If you loop over every element by hand, one by one, in the slowest way,
even a tiny model takes ages. And the algorithms downstream don't want to care *how* the
arithmetic is done — they just want a number back.

The bigger problem is the **heavy algebra**. Models routinely need things like the inverse
of a matrix, eigenvalues, or a singular-value decomposition — the "H-bombs" of linear
algebra. Writing those from scratch is both slow and error-prone. Professional libraries
have spent decades tuning them. But pulling in the traditional C/Fortran implementations
("system BLAS") means the library will not even *compile* on a machine without a C
toolchain — a heavy burden for a pure-Rust project.

So `sciencekit_math` had to solve two things at once:

- a **fast, reusable set of small operations** (the "kernels") written the Rust way — with
  zero copying and parallelism — that downstream algorithms call like building blocks; and
- a **single clean door** (an *interface*) through which all the heavy algebra passes, so
  that today it uses a pure-Rust implementation and tomorrow, if a user enables it, a
  blazing system BLAS — without changing a single line of algorithm code.

The rest of this chapter is the story of how those two pieces were built and why they look
the way they do.

---

## 2. The decisions — and the roads not taken

| Decision | Why we chose it | Alternatives we discarded |
|---|---|---|
| **Higher-order kernels via `azip!`/`par_azip!`, never manual index loops** | The PRD (§4.2) mandates higher-order iteration. Passing the operation as a *closure* (`|x| 2*x+1`) separates *what* to do from *how to walk the array*, and `par_azip!` gives us parallelism by changing one macro — the algorithm code never changes (spec `higher-order-kernels`). | Hand-written `for i in 0..len { ... }` loops everywhere — fast to write, but impossible to parallelize in one place and easy to get wrong. |
| **Squared-Euclidean distance via the norm expansion `‖a‖² − 2a·b + ‖b‖²`** | Expanding the square avoids a per-pair subtract loop, reuses a SIMD-friendly dot product over contiguous row memory, and is symmetric with a stable diagonal (spec `pairwise-distances`). | The literal `Σ (a_k − b_k)²` loop per pair — correct but slower and harder to vectorize. |
| **Pure-Rust `faer` as the default BLAS/LAPACK backend** | A pure-Rust implementation compiles anywhere with no C/Fortran toolchain, which the PRD requires as the default (spec `blas-interface`, Decision 1). `faer` compiles on the project's Rust 1.85. | `oxiblas` — a competing pure-Rust BLAS, rejected because it fails to compile on Rust 1.85. |
| **The `blas-backend` feature as an opt-in escape hatch** | Users who want the fastest possible system BLAS (OpenBLAS/BLIS/MKL/Accelerate) can enable a feature flag to swap `faer` for `ndarray-linalg`; `faer` stays available as a fallback (spec `blas-interface`). | Linking a system BLAS always — would make the default build fail on machines without one. |
| **Sparse products over zero-copy `CsMatView`, never densifying** | Lasso, linear SVMs and text classification operate on very sparse matrices (mostly zeros); copying them into dense form would waste memory (spec `dense-sparse-products`). The CSR×dense kernel walks only the non-zero entries. | Turning the sparse input into a dense array and using a normal multiply — simple but memory-hostile for sparse data. |

---

## 3. The concepts, taught from zero

This chapter needs a handful of Rust ideas and a handful of mathematical ones. Let's meet
them gently before touching any real code.

### Rust concept 1: closures — functions you can hand around

In Rust you can write a small function "in place," inline, right where you need it. Such an
anonymous function is called a **closure**. You have already seen them in this book's
earlier chapters (`|x| 2.0 * x + 1.0`). The key superpower of a closure is that you can
**pass it as an argument to another function**.

```rust
// `transform` is a parameter that is itself a function: it takes an F and returns an F.
pub fn sk_elementwise_transform<F: SKFloat>(
    input: &ArrayView<F, ndarray::Ix1>,
    transform: impl Fn(F) -> F + Sync + Send,
    parallelism: usize,
) -> Array<F, ndarray::Ix1>
```

Read the middle parameter type: `impl Fn(F) -> F + Sync + Send`. `Fn` is the Rust name for
"this is a callable thing," and `(F) -> F` says it takes one value of type `F` and returns
one. So `sk_elementwise_transform` is a *generic* worker: **it does not know what the
operation is** — the caller supplies the operation as a closure, and the worker applies it
to every element. This is the essence of a **higher-order function**: a function that
takes or returns other functions. It separates "what computation" (the closure, chosen by
the caller) from "how to sweep the array" (the kernel, written once). The `+ Sync + Send`
bits are promises that the closure can be shared across threads — which we need when we
parallelize.

### Rust concept 2: `azip!` and `par_azip!` — elementwise iteration done right

The whole library refuses to write `for i in 0..n` manual index loops for elementwise
work. Instead it uses the `ndarray` macros `azip!` and `par_azip!`, which iterate
element-by-element across one or more arrays **in lockstep**, element `0` with element `0`,
element `1` with element `1`, and so on. Look at the body of the transform:

```rust
azip!((out in &mut output, a in input) { *out = transform(*a); });
```

This says: "walk `input` and `output` together; for each pair, put `transform(*a)` into
`*out`." The macro handles the striding and indexing for you. `par_azip!` does the *same
thing in parallel* — it splits the work across a thread pool. Because both macros share the
exact same body syntax, you get parallelism almost for free: the only change between the
sequential and parallel versions is the macro name.

### Rust concept 3: owned arrays vs zero-copy views

An **owned** array, `Array<F, Ix2>`, *owns* its numbers in memory — it is the sole holder
and can resize, mutate, or be dropped. A **view**, `ArrayView2<F>`, does *not* own data; it
is a "window" that **borrows** someone else's owned array, remembering the shape and how to
reach each element. Views are the library's secret to speed: a function can take a view of
your existing buffer, do work, and hand you an answer — **without ever copying your data**.
Copying a matrix of a million rows is expensive; borrowing a view of it is essentially free.
That is why every public kernel here takes `&ArrayView2<F>` (a borrow of a view) and only
returns a fresh owned `Array2<F>` when it genuinely needs a new result.

### Rust concept 4: traits and a trait-object-free generic backend

A **trait** is a promise that a type can do certain things (an interface). Here the central
trait is `SKMathBackend`:

```rust
pub trait SKMathBackend<F: SKFloat>: Send + Sync {
    fn gemm(&self, a: ArrayView2<F>, b: ArrayView2<F>, alpha: F, parallelism: usize) -> Array2<F>;
    fn svd(&self, a: ArrayView2<F>) -> Result<SKSingularValueDecomposition<F>, SKError>;
    // ... more: qr, cholesky, solve, eigh, lu, ...
}
```

Any type that `impl`s this trait is a "math backend": it promises to perform GEMM (matrix
multiplication), SVD, QR, and so on. The trick here is that the trait is **generic over `F`
and the concrete backend type is known at compile time** — the caller writes
`let backend = SKFaerBackend::new();` and then calls `backend.gemm(...)`. Because the
concrete type is fixed, the compiler generates a direct, specialized call. This is the
opposite of a **trait object** (a `Box<dyn SKMathBackend>`), which would defer the choice to
runtime and pay for a dynamic dispatch on every call. For a performance crate, avoiding
that indirection — "trait-object-free" — is deliberate: heavy algebra runs on hot paths
where an extra virtual call per operation would add up.

### Numerical concept 1: what a "kernel" is

In numerical libraries a **kernel** is a small, reusable operation applied over an array or
matrix. We meet three families:

- **Elementwise transform** — a function applied to every element independently:
  `x → 2x + 1`. Input and output have the same shape.
- **Reduction** — collapse many values into fewer (often one): a *sum* turns a column into
  a single number.
- **In-place transform** — mutate the existing buffer rather than allocate a new one, saving
  memory.

The `sciencekit_math` kernels are the "LEGO bricks" that downstream algorithms snap
together.

### Numerical concept 2: what a distance matrix is, and squared-Euclidean math

A **distance matrix** is a table where entry `[i, j]` is "how far apart are row `i` of the
first set and row `j` of the second set." Each row of a matrix is a *point* in
multi-dimensional space (here, one point per flower).

The **Euclidean distance** between two points `a` and `b` is the length of the straight
line between them: `√(Σ (a_k − b_k)²)`. Its square is `Σ (a_k − b_k)²`. Computing that
literally means subtracting each coordinate pair. But algebra gives us a shortcut by
expanding the square:

```
‖a − b‖² = Σ (a_k − b_k)² = Σ (a_k² − 2·a_k·b_k + b_k²) = ‖a‖² − 2·(a·b) + ‖b‖²
```

where `‖a‖² = Σ a_k²` (the squared "length" of `a`) and `a·b = Σ a_k·b_k` is the **dot
product** of `a` and `b`. Why bother? Because we can precompute `‖a‖²` for *every* row once,
and `‖b‖²` for every row of the other set once, and then each cell only needs one dot
product — a tight, SIMD-friendly loop over contiguous memory. This is the norm-expansion
formulation the spec mandates, and it is the whole reason `sciencekit_math` computes
distances this way.

### Numerical concept 3: cosine and Manhattan distances

**Cosine distance** measures the *angle* between two points rather than their distance.
The cosine of the angle between `a` and `b` is `(a·b) / (‖a‖·‖b‖)`. If the vectors point the
same way the cosine is `1`; if they are perpendicular (orthogonal) it is `0`. Distance is
then defined as `1 − cosine`, so identical directions score `0` and orthogonal directions
score `1`. Nice property: cosine distance is **scale-invariant** — `[1,2]` and `[10,20]`
point the same way, so their cosine distance is `0`.

**Manhattan distance** (also called **L1**) is the distance you would walk along a city
grid — only left/right and up/down, no diagonals: `Σ |a_k − b_k|`. It is the sum of the
absolute coordinate differences.

### Numerical concept 4: sparse data and the CSR format

A matrix is **sparse** when most of its entries are zero — think of text classification,
where each row is a document and each column says "how many times word *X* appeared." Storing
all those zeros as real numbers wastes memory and time. The **CSR** (Compressed Sparse Row)
format stores only the non-zero values, in three parallel lists:

- the non-zero **values** themselves,
- the **column** each value belongs to,
- a **pointer array** that says where each *row* begins.

To multiply a CSR matrix by a dense matrix, you only touch the non-zero entries — no need
to ever build a big dense matrix full of zeros. `sciencekit_math` consumes CSR through a
view (`CsMatView`) so even the sparse input is never copied.

### Numerical concept 5: C vs F memory contiguity

A matrix is stored in memory as a long list of numbers. **Row-major (C) order** lays out
each row back-to-back; **column-major (F) order** lays out each column back-to-back. Most
languages default to row-major (the "C" convention); Fortran-derived libraries default to
column-major ("F"). For a hot loop it matters *a lot* which layout a buffer has, because
CPUs love reading memory that is sequential. A contiguous buffer (all elements one after
another) is much faster to sweep with SIMD than a strided one. So the layout helpers detect
which case we are in and, when a kernel needs contiguous memory, produce a contiguous copy
*only when necessary* — borrowing when already contiguous (zero copy) and owning a fresh copy
only when strided.

### Numerical concept 6: what BLAS and LAPACK are

**BLAS** (Basic Linear Algebra Subprograms) is a de-facto standard set of routines for
vector and matrix arithmetic — the crown jewel being **GEMM**, the general matrix
multiplication `C = α·A·B`. **LAPACK** builds on BLAS to provide the higher-level
decompositions: SVD, QR, Cholesky, LU, eigenvalues, least squares. For decades these have
been hand-tuned in C/Fortran (OpenBLAS, Intel MKL, and so on) to be staggeringly fast. The
catch: they need a native C toolchain to compile. `sciencekit_math` solves this by defining
its own `SKMathBackend` interface and letting the *default* implementation be **`faer`**, a
modern pure-Rust BLAS/LAPACK — fast, and compiles anywhere Rust does.

---

## 4. Each object, explained

### `SKMathBackend` (the dense algebra trait)

*What it is:* The single host-centric interface for all heavy dense algebra. It is a trait
generic over `SKFloat`, whose methods take zero-copy `ArrayView2` inputs and return owned
`Array2` results. The surface spans `gemm`, `svd`, `qr`, `cholesky`, `solve_triangular`,
`solve`, `eigh`, `lu`, `slogdet`, `pinv`, `inv`, `lstsq`, and `norm`/`vector_norm`. It also
has *default implementations*: for example `slogdet` is derived from LU, `pinv` and `lstsq`
are composed from an SVD, and `norm` reduces host-side — so a backend can implement only the
primitives and still get everything else for free.

*Why it exists:* Downstream algorithm crates must not care *which* library does the heavy
algebra. They call methods on the trait; swapping `faer` for `ndarray-linalg` (under the
`blas-backend` feature) is a config change, not a rewrite (spec `blas-interface`). The
zero-copy views on input keep the promise of not copying the caller's data.

*What we rejected:* Letting a concrete backend type leak into the algorithm surface, and
using a trait object (`Box<dyn ...>`) — the latter would add runtime dispatch on hot
numerical paths, which the performance mandate rules out.

### `SKFaerBackend` (the pure-Rust default backend)

*What it is:* A tiny, `Debug + Clone + Copy + Default` unit struct — it holds no data —
created with `SKFaerBackend::new()`. It implements `SKMathBackend<f64>` and
`SKMathBackend<f32>` by delegating GEMM to faer's blocked `matmul` (honouring the plan's
parallelism) and the decompositions to faer's high-level SVD/QR/Cholesky/LU/eigen solvers.
A sibling `SKMatrixMultiplyBackend` routes GEMM to `matrixmultiply` and the decompositions
to faer.

*Why it exists:* The default must work with no C toolchain. faer compiles on Rust 1.85 and
is competitive, so the default build links no system BLAS at all.

*What we rejected:* Making a system BLAS the default, and `oxiblas` (fails to compile on the
project's Rust version). The `unsafe` here is contained in the SIMD/GEMM glue only.

### `sk_elementwise_transform` (the elementwise kernel)

*What it is:* A free function that applies a closure `x → transform(x)` to every element of
a 1-D view and returns a new owned array. It reads the resolved `parallelism`: `1` runs the
sequential `azip!` form (zero dispatch overhead); `> 1` runs `par_azip!` across the compute
pool.

*Why it exists:* It is the most basic LEGO brick — every scaling, normalization or feature
transform reuses it (spec `higher-order-kernels`). Iteration is delegated to `azip!`, never
a manual index loop.

*What we rejected:* Hand-written index loops, and always-owning a fresh buffer when the
caller might prefer an in-place variant (that is `sk_scale_in_place`'s job instead).

### `sk_binary_combine` (the pairwise elementwise kernel)

*What it is:* A free function that pairs two equal-length views element by element,
`combine(left_k, right_k)`, and returns a new owned array. It asserts the lengths match and,
like the transform, honours `parallelism` (`1` → `zip_mut_with`, `> 1` → `par_azip!`).

*Why it exists:* Many operations need two arrays at once — adding a bias term, multiplying
two feature sets, comparing two signals. It is the binary sibling of the transform.

*What we rejected:* Requiring a pre-allocated output buffer from the caller; the function
owns its result, keeping call sites simple.

### `sk_axis_sum` (the reduction kernel)

*What it is:* A free function that sums a 2-D array along one axis into a 1-D result.
`axis = 0` collapses rows → per-column sums (length = number of columns); `axis = 1`
collapses columns → per-row sums (length = rows). It honours `parallelism`. The parallel
axis-0 path splits rows into chunks, reduces each chunk into its *own* accumulator, then
combines the partials — deliberately avoiding any shared mutable accumulator, so there is
no data race.

*Why it exists:* Summing along an axis is everywhere (totals, statistics, normalization
denominators). The race-free parallel design is the careful part (spec `higher-order-kernels`).

*What we rejected:* Writing directly into a single shared accumulator from multiple threads
(that would race), and manual index loops for the reductions.

### `sk_scale_in_place` (the in-place kernel)

*What it is:* A free function that multiplies every element of a 2-D array by a scalar
factor **in place**, returning nothing. It mutates the caller's buffer, allocating no new
array (spec `higher-order-kernels`).

*Why it exists:* When you only need to scale existing data, copying the whole array is
wasteful; in-place is the zero-allocation path.

*What we rejected:* Returning a new array for scaling (that is `sk_elementwise_transform`'s
mode); this function's whole point is to avoid the copy.

### `SKMemoryLayout`, `sk_memory_layout`, `sk_force_contiguous` (the layout helpers)

*What they are:* `SKMemoryLayout` is a three-variant enum — `CContiguous` (row-major),
`FContiguous` (column-major), `Strided` (neither). `sk_memory_layout` detects which case a
view is in; `sk_is_c_contiguous` / `sk_is_f_contiguous` ask one yes/no question; and
`sk_force_contiguous` returns a `CowArray`: a zero-copy borrow when the input is already
contiguous, or a fresh owned contiguous copy when it is strided.

*Why they exist:* Hot SIMD kernels want contiguous memory. Detecting layout lets code feed
fast paths with contiguous buffers and only pay for a copy when it truly has to.

*What we rejected:* Always copying (defeats zero-copy), and assuming a layout that the data
does not have (would silently run slow or wrong).

### The distance functions: `sk_squared_euclidean_distance_matrix`, `sk_euclidean_distance_matrix`, `sk_manhattan_distance_matrix`, `sk_cosine_distance_matrix`

*What they are:* Four free functions that take two `&ArrayView2<F>` feature matrices and
return an owned `Array2<F>` distance matrix. Squared-Euclidean uses the norm-expansion
`‖a‖² − 2a·b + ‖b‖²`, with row norms precomputed and each cell needing one SIMD dot product;
Euclidean is its element-wise square root; Manhattan is `Σ |a_k − b_k|`; cosine is
`1 − (a·b)/(‖a‖·‖b‖)`, with a guard that maps zero-norm rows to distance `1` (matching
scikit-learn).

*Why they exist:* Nearest-neighbor search, clustering, SVM kernels and imputation all need
distance matrices. The norm-expansion formulation is the performance heart (spec
`pairwise-distances`).

*What we rejected:* The naive per-pair subtract loop `Σ (a_k − b_k)²` for squared Euclidean
(slower, not SIMD-friendly), and dividing by zero in cosine without a guard.

### `sk_csr_dense_product` and `sk_sparse_product` (the sparse products)

*What they are:* `sk_csr_dense_product` multiplies a `CsMatView` by a dense `ArrayView2`,
walking each non-zero entry of the CSR operand once and accumulating `value * dense_row`
into the output row via `azip!` — the sparse operand is **never densified**. `sk_sparse_product`
multiplies two `CsMatView<f64>` and delegates to sprs' sparse matrix-matrix product (SMMP),
returning a sparse `CsMat<f64>`.

*Why they exist:* Sparse workloads (Lasso, linear SVMs, text classification) would blow up
memory if densified (spec `dense-sparse-products`). Both take zero-copy views.

*What we rejected:* Densifying the sparse operand before multiplying, and copying the sparse
input.

---

## 5. A real walk-through, using the tests

The most convincing way to learn is to read actual tests that passed. Here are three real
tests: one from `kernels_tests.rs`, one from `pairwise_tests.rs`, and one that proves the
parallel and sequential paths agree.

### Test 1: `axis_one_sum_matches_row_totals` (from `kernels_tests.rs`)

This test checks that `sk_axis_sum` with `axis = 1` sums each row correctly.

```rust
#[test]
fn axis_one_sum_matches_row_totals() {
    let input: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let sums = sk_axis_sum(&input.view(), 1, 1);
    assert_eq!(sums, array![6.0, 15.0, 24.0]);
}
```

Line by line:

1. `let input: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];` — we
   build an owned 3×3 matrix of `f64`. The `array!` macro lays out three rows, each with
   three numbers, so `input` has 3 rows and 3 columns.
2. `let sums = sk_axis_sum(&input.view(), 1, 1);` — we call the kernel. The `&input.view()`
   hands the function a *view* (a borrow, no copy). The first `1` is the axis: we are summing
   along axis 1, which collapses columns and keeps rows. The second `1` is the parallelism:
   run sequentially.
3. `assert_eq!(sums, array![6.0, 15.0, 24.0]);` — we compare to the hand-computed row totals.
   Row `[1,2,3]` sums to `6`, row `[4,5,6]` to `15`, row `[7,8,9]` to `24`. The kernel's
   sequential path iterates rows and folds each one, so the output is exactly this.

### Test 2: `squared_euclidean_matches_direct_computation` (from `pairwise_tests.rs`)

This test proves the norm-expansion squared-Euclidean kernel against hand-computed values.

```rust
#[test]
fn squared_euclidean_matches_direct_computation() {
    let left: Array2<f64> = array![[0.0, 0.0], [3.0, 4.0]];
    let right: Array2<f64> = array![[1.0, 1.0]];
    let distance = sk_squared_euclidean_distance_matrix(&left.view(), &right.view());
    // dist([0,0],[1,1]) = 2 ; dist([3,4],[1,1]) = 4 + 9 = 13.
    assert!((distance[[0, 0]] - 2.0).abs() < 1e-12);
    assert!((distance[[1, 0]] - 13.0).abs() < 1e-12);
}
```

Line by line:

1. `let left: Array2<f64> = array![[0.0, 0.0], [3.0, 4.0]];` — the first feature set has two
   points: `(0,0)` and `(3,4)`. `let right: Array2<f64> = array![[1.0, 1.0]];` — the second
   set has one point `(1,1)`. So the distance matrix will be 2×1.
2. `let distance = sk_squared_euclidean_distance_matrix(&left.view(), &right.view());` — we
   hand both sets in as views. Inside, the kernel computes `‖left_row‖²` for both rows
   (`0` and `25`), `‖right_row‖²` for the single right row (`2`), then fills the 2×1 matrix
   with `‖a‖² + ‖b‖² − 2·(a·b)` per cell.
3. `assert!((distance[[0, 0]] - 2.0).abs() < 1e-12);` — the distance from `(0,0)` to `(1,1)`
   is `0 + 2 − 2·(0·1+0·1) = 2`, matching the straight-line square `(1−0)² + (1−0)² = 2`.
   The `.abs() < 1e-12` is a floating-point tolerance: we expect exactly `2`, allowing tiny
   rounding.
4. `assert!((distance[[1, 0]] - 13.0).abs() < 1e-12);` — from `(3,4)` to `(1,1)` the squared
   distance is `(1−3)² + (1−4)² = 4 + 9 = 13`, which the norm-expansion also produces.

### Test 3: `elementwise_parallel_matches_sequential_on_large_array` (from `kernels_tests.rs`)

This test checks that the parallel kernel agrees with the sequential one, which is the whole
point of the `parallelism` switch.

```rust
#[test]
fn elementwise_parallel_matches_sequential_on_large_array() {
    let n = 1 << 16;
    let input: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64) / 7.0);
    let f = |x: f64| (x * 3.0 + 2.0).sin();
    let sequential = sk_elementwise_transform(&input.view(), f, 1);
    let parallel = sk_elementwise_transform(&input.view(), f, 8);
    assert_close(&sequential, &parallel);
}
```

Line by line:

1. `let n = 1 << 16;` — `n` is `2¹⁶ = 65536`, a large array so that parallelism actually
   matters.
2. `let input: Array1<f64> = Array1::from_shape_fn(n, |i| (i as f64) / 7.0);` — we build a
   1-D array of 65536 elements; element `i` is `i / 7.0`. `from_shape_fn` fills the shape by
   calling the closure for each index.
3. `let f = |x: f64| (x * 3.0 + 2.0).sin();` — the transform: `sin(3x + 2)`. Note the same
   closure is passed to both calls; this is the higher-order design in action.
4. `let sequential = sk_elementwise_transform(&input.view(), f, 1);` — run it sequentially
   (`parallelism = 1`).
5. `let parallel = sk_elementwise_transform(&input.view(), f, 8);` — run the *same* transform
   with `parallelism = 8`, which dispatches `par_azip!` across the pool.
6. `assert_close(&sequential, &parallel);` — the test's own helper checks the two agree within
   a relative tolerance. Why tolerance instead of exact equality? Parallel reordering can
   shift the *last bits* of floating-point results, so we allow tiny differences. This is the
   contract that parallel kernels stay correct.

---

## 6. Look inside

<div class="sk-box sk-box--tip">
  <p><strong>One strong insight:</strong> the crate's performance is an <em>act of
  architecture</em>, not just clever loops. The kernels separate <em>what</em> to compute (a
  closure) from <em>how</em> to sweep memory (`azip!`/`par_azip!`), distances are rewritten
  into a form that reuses one SIMD dot product per cell, sparse data is never densified, and
  the heavy algebra hides behind a generic trait so the default is pure Rust and the fast
  path is an opt-in flag — all without a single algorithm ever changing.</p>
</div>

Here is how the crate's modules hang together:

```mermaid
flowchart LR
    MATH[`sciencekit_math`] --> K[kernels]
    MATH --> L[layout]
    MATH --> PW[pairwise]
    MATH --> SO[sparse_ops]
    MATH --> BE[backend]

    K --> K1[sk_elementwise_transform]
    K --> K2[sk_binary_combine]
    K --> K3[sk_axis_sum]
    K --> K4[sk_scale_in_place]
    K --> AZ["azip! / par_azip! (higher-order iteration)"]

    L --> L1[SKMemoryLayout C|F|Strided]
    L --> L2[sk_force_contiguous -> CowArray]

    PW --> P1[sk_squared_euclidean_distance_matrix]
    PW --> P2[sk_euclidean_distance_matrix]
    PW --> P3[sk_manhattan_distance_matrix]
    PW --> P4[sk_cosine_distance_matrix]
    P1 --> SIMD["wide f64x4 SIMD dot (simd_dot.rs)"]

    SO --> S1[sk_csr_dense_product (never densifies)]
    SO --> S2[sk_sparse_product -> sprs SMMP]

    BE --> TRAIT[SKMathBackend trait]
    TRAIT --> F[SKFaerBackend (pure Rust, default)]
    TRAIT --> G[SKMatrixMultiplyBackend]
    TRAIT --> H["ndarray-linalg (opt-in blas-backend)"]

    style MATH fill:#e8f4f8,stroke:#6c8ebf
    style K fill:#fff2cc,stroke:#d6b656
    style L fill:#fff2cc,stroke:#d6b656
    style PW fill:#e8f5e9,stroke:#7ea6a0
    style SO fill:#e8f5e9,stroke:#7ea6a0
    style BE fill:#fce4ec,stroke:#c2185b
    style TRAIT fill:#fce4ec,stroke:#c2185b
```

The diagram shows `sciencekit_math` as the numeric substrate: the four module families
(`kernels`, `layout`, `pairwise`, `sparse_ops`) supply the reusable operations, all built on
`azip!`/`par_azip!` and zero-copy views, while `backend` supplies the single heavy-algebra
door that defaults to pure-Rust `faer` and can be swapped for a system BLAS via a feature
flag.

---

## 7. Recap

- **Kernels separate *what* from *how*.** `sk_elementwise_transform` and `sk_binary_combine`
  take a closure (the operation) and sweep memory with `azip!`/`par_azip!` — so switching to
  parallel is a one-word change, never a rewrite.
- **Reductions are race-free by design.** `sk_axis_sum` accumulates per-chunk partials in
  parallel rather than writing into one shared accumulator, so parallel results match the
  sequential ones within a tiny floating-point tolerance.
- **Distances use math to win.** Squared-Euclidean is computed as `‖a‖² − 2a·b + ‖b‖²`,
  reusing a SIMD dot product per cell; cosine and Manhattan have their own clean formulas,
  all over zero-copy row views.
- **Sparse stays sparse.** `sk_csr_dense_product` walks only the non-zero entries and never
  densifies, while `sk_sparse_product` delegates to sprs' SMMP.
- **Heavy algebra hides behind one trait.** `SKMathBackend` with the pure-Rust `SKFaerBackend`
  default, plus an opt-in `blas-backend` feature for a system BLAS — the default needs no C
  toolchain, and algorithms never change when the backend does.

*Next chapter:* armed with kernels, distances, sparse products and a linear-algebra door, we
are ready to watch the first real estimator put them to work — fitting a model and making
predictions on new data.