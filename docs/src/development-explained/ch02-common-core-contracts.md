# Ch. 2 — Core contracts in `sciencekit_common`

Before any algorithm can crunch numbers, the whole library has to agree on a shared
*vocabulary*: what a number is, what an error looks like, what a piece of data is, and
what "fit" or "predict" even mean as a promise. That shared vocabulary lives in one
crate, `sciencekit_common`. This chapter walks through the foundational traits and types
every future algorithm will compile against — starting from a reader who has never seen
Rust generics, traits, or even floating-point numbers.

---

## 1. The problem we were solving

Imagine a whole team of engineers each building a different algorithm: one builds a
linear regression, another a decision tree, another a clustering method. If they each
invent their own way to represent the data, their own error messages, and their own idea
of what "fit" means, the library becomes a zoo — every algorithm speaks a different
dialect, and nothing composes with anything else.

Rust makes this worse because of its philosophy of **strict types**. The compiler
refuses to let two functions interoperate unless their types line up exactly. So the very
first thing a library like this needs is not an algorithm at all — it is a **contract**:
a set of shared building blocks that every future piece of code can agree on.

`sciencekit_common` is that contract. It is the crate that "freezes the vocabulary every
future crate compiles against" (as its own `lib.rs` header says). It decides:

- what a valid floating-point number is (`SKFloat`),
- how failures are reported (`SKError`),
- how feature data and target data enter an algorithm (`SKDataView`, `SKTargetView`),
- what it means for an algorithm to *fit*, *predict*, or *transform*
  (`SKSupervisedFit`, `SKUnsupervisedFit`, `SKPredictor`, `SKFeatureTransformer`).

Without this crate, `sciencekit` would be dozens of disconnected experiments. With it,
every algorithm starts from a proven, shared foundation — like every room in a building
sharing the same electrical standard and the same wall sockets.

Think of it as a plug-and-socket problem. Every appliance (algorithm) needs the same
plug shape (the data-view contract), the same socket voltage (the scalar type), and the
same way to tell you it broke (the error type). `sciencekit_common` manufactures those
plugs, sockets, and fuses once, so every appliance can share a single wall.

---

## 2. The decisions — and the roads not taken

| Decision | Why we chose it | Alternatives we discarded |
|---|---|---|
| **One sealed `SKFloat` trait** for continuous numbers | Every generic algorithm can accept `f32` or `f64` through a single bound, but external crates cannot invent their own numeric types (spec `scalar-typing`). The private seal module enforces "only the floats we chose" at compile time. | Loose bounds like `Float + Send + Sync` scattered everywhere — allows any third-party to satisfy the contract, which is risky and inconsistent. |
| **Enums (`SKDataView`, `SKTargetView`) marked `#[non_exhaustive]`** | Dense vs sparse features and continuous vs integer vs nominal targets fit naturally as an either-or choice; `non_exhaustive` lets us add variants later without breaking consumers who match on the enum. | Separate trait objects or several distinct function signatures per representation — more code, less uniform, harder to extend. |
| **Zero-copy views, not owned copies, at public inputs** | The whole point of the library is speed; borrowing the caller's existing buffer (via `ArrayView2`/`CsMatView`) avoids duplicating huge matrices (spec `data-view-boundary`). | Passing owned `Array2` by value forces a copy and forces the caller to give up their data. |
| **Public inputs over the `TryInto` conversion seam** | Any type that can convert into a `SKDataView` (including third-party types) is accepted automatically; fallible conversions report a structured `SKError` instead of panicking. | Fixing the API to only accept `ndarray`/`sprs` types — locks out external integrators. |
| **Central `SKError` enum + per-algorithm error via `From`** | One precise taxonomy (shape, unsupported representation, hyperparameter, convergence, I/O, conversion) shared everywhere; algorithms still get their own error type that converts from it automatically (spec `error-model`). | Ad-hoc error types per algorithm, or a single `String` error — either loses precision or loses consistency. |

---

## 3. The concepts, taught from zero

This chapter leans on several Rust ideas and several numerical ideas. Let's meet each
before we look at real code.

### Rust concept 1: what a trait is

A **trait** in Rust is like an interface in other languages: a promise that a type can do
certain things. When we write

```rust
trait SKFloat: Float { /* ... */ }
```

we are saying: *"anything that claims to be an `SKFloat` must also be a `Float`"* — that
is, it must support arithmetic like `+`, `*`, `.abs()`, and so on. Types then **implement**
the trait:

```rust
impl SKFloat for f32 {}
impl SKFloat for f64 {}
```

This says "the types `f32` and `f64` promise to be `SKFloat`."

### Rust concept 2: generics — writing one function for many types

A **generic** function is written once but works on many types. In many languages you
write `template <typename T>`; in Rust you write `<T>` or, in our case, `<F: SKFloat>`.
The colon is a **bound**: "F can be *any* type, as long as it satisfies `SKFloat`."

```rust
fn scale<F: SKFloat>(value: F, factor: F) -> F {
    value * factor
}
```

This function does not care whether `F` turns out to be `f32` or `f64` — it just knows
that whatever `F` is, it supports `*` and can be returned. The caller chooses the type:

```rust
let x32 = scale(2.0_f32, 3.0_f32); // F is f32
let x64 = scale(2.0_f64, 3.0_f64); // F is f64
```

Both call the *same* function, but the compiler produces a specialized version for each
concrete type. This is called **monomorphization**, and it is why generics cost nothing at
runtime — there is no "boxing" or dynamic lookup; it is just ordinary statically typed
code, duplicated per type.

### Rust concept 3: sealed traits (why only `f32`/`f64` can be `SKFloat`)

A **sealed trait** is a trait that outsiders cannot implement. We want `SKFloat` to be
implemented *only* by `f32` and `f64` — not by some future third-party numeric type, and
certainly not by integers. Rust has no built-in "seal" keyword, so we achieve sealing with
a clever trick: make the trait **require a private supertrait** that only we can
implement.

```rust
pub trait SKFloat: Float + Send + Sync + 'static + private::SKFloatSealed {}
```

The `private::SKFloatSealed` supertrait lives in a `private` module that is *not* exported.
Because no outside crate can even name that module, no outside crate can write
`impl SKFloatSealed for MyType`. And since implementing `SKFloat` requires implementing
`SKFloatSealed`, only the two implementations we wrote (for `f32` and `f64`) can ever
exist. The sealing is enforced at **compile time**, by the module system itself.

This is subtle and powerful: the constraint is not checked at runtime — it *cannot even be
written* by anyone outside the crate. That is exactly the safety we want.

### Rust concept 4: enums — a value that is one of several shapes

An **enum** is a type whose value is *exactly one* of a fixed set of possibilities. In
Rust each possibility can carry its own data. For example:

```rust
pub enum SKDataView<'a, F> {
    Dense(ArrayView2<'a, F>),   // a 2-D dense matrix view
    Sparse(CsMatView<'a, F>),   // a sparse matrix view
}
```

A `SKDataView` value is *either* a dense view *or* a sparse view — never both. When we
later `match` on it, we write one branch per case and the compiler guarantees we handled
every possibility. `#[non_exhaustive]` means "there may be more variants in the future, so
don't write a `match` without a wildcard `_` arm."

### Rust concept 5: associated types (`type Model`)

Some traits promise not just methods but also a **type** that is chosen by the implementer.
This is an **associated type**:

```rust
pub trait SKSupervisedFit<F: SKFloat> {
    type Model;   // the fitted model type, chosen by the implementer
    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>;
}
```

Whoever implements `SKSupervisedFit` must say what `Model` is. When a linear regression
implements this trait, its `Model` is a fitted-linear-regression struct; when a decision
tree does, its `Model` is a fitted-tree struct. The *contract* of `fit` is fixed, but the
concrete fitted type is up to each algorithm. `type Error` is the same idea for errors:
each algorithm may pick its own error type, as long as it can be built `From` the central
`SKError`.

### Rust concept 6: zero-copy views vs owned arrays

This is both a Rust and a performance concept. An **owned array** (`Array2<f64>`) owns its
data in memory — it is the actual box of numbers. A **view** (`ArrayView2<'a, f64>`) is
like a window that *looks at* somebody else's numbers without owning them. The view is
tiny (a few pointers and length fields), and it **borrows** the underlying buffer.

The `'a` is a **lifetime**: it says "this view is only valid for as long as the data it
borrows is alive." Rust's borrow checker guarantees you never hold a view to data that has
been freed. The `lib.rs` crate even forbids `unsafe_code` at the crate level
(`#![forbid(unsafe_code)]`), relying entirely on safe Rust and the borrow checker for
memory safety.

Creating a view is free — no elements are copied. Copying a big matrix element-by-element
is exactly the kind of waste this library was built to avoid.

### Rust concept 7: the `TryInto` conversion seam

Rust's standard library provides two conversion traits:

- `Into<T>` — an **infallible** conversion that cannot fail.
- `TryInto<T>` — a **fallible** conversion that returns a `Result`, so it can fail.

The key insight in `sciencekit` is that public APIs declare their inputs over `TryInto`,
not over a fixed concrete type. So `fit` does not say "give me an `Array2`"; it says
"give me *anything* that can `TryInto` a `SKDataView`":

```rust
where
    X: TryInto<SKDataView<'a, F>, Error = SKError>,
```

Because `f64: Into<T>` is automatically also `TryInto<T>` (a blanket rule in the standard
library), an infallible conversion is accepted for free. And a third-party library can
integrate by writing its own `impl TryFrom<TheirType> for SKDataView`. This makes the
library **open to extension** at its boundaries without ever changing the core.

### Rust concept 8: error handling via `Result` + `thiserror`

Rust functions that can fail return a `Result<T, E>` — either `Ok(value)` on success or
`Err(error)` on failure. The `thiserror` crate makes defining good error types easy: you
write the enum, and `#[derive(Error)]` plus `#[error("...")]` attributes generate the
`Display` and `source` implementations for you. That is why our error enum is so readable.

### Numerical concept 1: what a floating-point scalar is

Numbers like `3.14` or `-0.001` with fractional parts are **floating-point** numbers. A
computer stores them in binary as sign + exponent + mantissa (scientific notation in base
2). They are *approximate* — not every decimal fraction can be represented exactly. The
name "float" comes from the decimal point being able to "float" to any exponent.

### Numerical concept 2: why `f32` vs `f64`

- `f32` is **32-bit single precision**: smaller (4 bytes each), faster on some hardware,
  but stores about 7 significant decimal digits.
- `f64` is **64-bit double precision**: larger (8 bytes each), slower, but stores about
  15–16 significant decimal digits — the standard for scientific computing.

Machine learning often defaults to `f64` for correctness in training (where repeated
arithmetic accumulates error) and can use `f32` for memory/throughput when accuracy is
good enough. Our sealed `SKFloat` allows *both*, so a single algorithm works with either —
the caller picks, and the code is generic over both. Integers, by contrast, are **not**
allowed here: continuous algorithms need fractional values and decimal precision, which
integers fundamentally lack (spec `scalar-typing`).

### Numerical concept 3: what a "view" over data means, and why zero-copy matters

A **view** over data is a lens, not a copy. If you have a 10-million-element matrix and you
pass a *view* to a function, the function sees the same underlying numbers without the
cost of duplicating 10 million elements. This is the heart of **zero-copy**: no bytes are
moved; only a lightweight descriptor travels.

Why does zero-copy matter for performance? Copying memory is slow relative to arithmetic.
In a pipeline of many stages (transform → scale → train), if every stage copied its input,
a matrix would be duplicated over and over — each copy costing bandwidth and time. By
borrowing views all the way through, the data stays in place and only descriptors move.
On very large, possibly out-of-core datasets, avoiding copies is the difference between
"fits in memory" and "swaps."

### Numerical concept 4: dense vs sparse (CSR) matrices

A **dense matrix** stores every cell, including all the zeros, in a regular grid. It is
simple and fast when most cells are nonzero.

A **sparse matrix** stores only the nonzero entries plus enough bookkeeping to remember
where they were. **CSR** stands for *compressed sparse row*: the matrix is stored row by
row, keeping (a) the values, (b) which column each value lives in, and (c) pointers marking
where each row begins. If 99% of a matrix is zeros (think user–item ratings, text, networks),
storing everything densely wastes memory. Sparse storage packs the same data far more
compactly.

A `SKDataView` can be either — an algorithm that works on both gets one uniform type to
match on, instead of two code paths.

### Numerical concept 5: what a "target" or "label" is in machine learning

In **supervised** learning you have features (the input, e.g. pixel values, measurements)
and a **target** (what you want to predict — the answer). Three flavors matter here:

- **Continuous**: a real number you predict, e.g. house price → `f64` (regression).
- **Integer**: a whole-number target, e.g. star ratings → `i64` (can be elevated to
  continuous losslessly).
- **Nominal**: a category, e.g. `"cat"`, `"dog"`, `"bird"` → strings (classification).

`SKTargetView` is the single type that holds any of these three. Unsupervised algorithms
(e.g. clustering) have no targets at all — which is exactly why `SKUnsupervisedFit::fit`
takes only features.

---

## 4. Each object, explained

### `SKFloat` (the sealed floating-point bound)

*What it is:* The single trait that says "this type is a valid continuous number for
`sciencekit`." It is `Float + Send + Sync + 'static + private::SKFloatSealed`, implemented
only for `f32` and `f64`. `Send + Sync + 'static` mean the value can be moved across
threads and lives forever (needed because algorithms may run in parallel).

*Why it exists:* Every generic algorithm needs a numeric bound, and having exactly one
guarantees consistency (spec `scalar-typing`). `Send + Sync + 'static` ensure a value of
type `F` is thread-transferable, which future parallel algorithms depend on.

*What we rejected:* A scatter of loose bounds on every function, and allowing third-party
numeric types. Sealing locks the surface so only our two floats are valid.

### `private::SKFloatSealed` (the private seal — non-public but essential)

*What it is:* A hidden supertrait, `#[doc(hidden)]`, defined in a `private` module, with
implementations only for `f32` and `f64`.

*Why it exists:* It is the mechanism that seals `SKFloat`. Because the module is private,
outsiders cannot name the trait, so they cannot implement it, and therefore cannot
implement `SKFloat`. This is the compile-time guarantee behind the whole scalar contract.

*What we rejected:* Language-level `sealed` keywords don't exist in Rust; this module trick
is the idiomatic, documented way to achieve sealing.

### `SKError` (the central error taxonomy)

*What it is:* A `#[non_exhaustive]` enum, derived with `thiserror`, with eight variants:
`ShapeMismatch`, `UnsupportedRepresentation`, `InvalidHyperparameter`,
`ExecutionModeIncompatible`, `BatchBufferOversize`, `NotConverged`, `Io`, and `Conversion`.
Each carries precise data (e.g. expected/found shapes, iteration counts, byte counts) and a
readable `Display` message.

*Why it exists:* One shared taxonomy means the same failure means the same error everywhere
(spec `error-model`). The `#[from] io::Error` variant lets platform I/O errors convert in
automatically; `#[non_exhaustive]` lets us grow the enum without breaking consumers.

*What we rejected:* Ad-hoc error types per algorithm and plain `String` errors — both lose
precision or consistency. Convenience constructors like `SKError::shape_mismatch_2d` and
`SKError::not_converged` exist to keep call sites tidy.

### `SKDataView` (the zero-copy feature view)

*What it is:* A `#[non_exhaustive]` enum over the two feature representations:
`Dense(ArrayView2<'a, F>)` and `Sparse(CsMatView<'a, F>)`. Both are *views* — they borrow
the caller's data. It derives `Debug, Clone, Copy`.

*Why it exists:* One uniform type for "the input features," whether dense or sparse,
enables a single dispatch per operation (spec `data-view-boundary`). The `.representation()`
method returns `"dense"`/`"csr"` for one-time dispatch and diagnostics; `.as_dense()`
returns the dense view or rejects a sparse input *before processing any element*.

*What we rejected:* Passing owned arrays (forces copies) and separate functions per
representation (duplication, no uniformity). `#[non_exhaustive]` keeps room for future
variants.

### `SKTargetView` (the zero-copy target view)

*What it is:* A `#[non_exhaustive]` enum over three target representations:
`Continuous(ArrayView1<'a, f64>)`, `Integer(ArrayView1<'a, i64>)`, and
`Nominal(&'a [&'a str])`. Also derives `Debug, Clone, Copy`.

*Why it exists:* Supervised algorithms need a single type for "the answers" that covers
regression (continuous), integer targets, and classification (nominal). `.as_continuous()`
elevates continuous or integer targets to `f64` **losslessly** (integers in the f64-
representable range) and rejects nominal with a clear suggestion.

*What we rejected:* Separate per-kind target arguments (forces the caller to know the kind
up front) and forcing nominal text to be copied (we borrow the strings instead — spec
`data-view-boundary`).

### `SKSupervisedFit` (the supervised fit contract)

*What it is:* A trait with associated `Model` and `Error` types whose `fit(&self, features, targets)`
takes *both* features and targets, each declared over the `TryInto` seam, and returns
`Result<Self::Model, Self::Error>`.

*Why it exists:* It is the promise every supervised estimator keeps: *given features and
targets, produce a fitted model* (spec `estimator-contracts`). Taking `&self` keeps the
configured estimator unchanged and reusable.

*What we rejected:* Returning the learned state on the estimator itself (that would let
you predict before fitting and prevent concurrent fits). The model is a **separate type**.

### `SKUnsupervisedFit` (the unsupervised fit contract)

*What it is:* A trait with associated `Model` and `Error` types whose `fit(&self, features)`
takes *only* features and returns `Result<Self::Model, Self::Error>`.

*Why it exists:* Clusterers and other unsupervised algorithms have no targets; making the
signature lack targets makes it *impossible* to pass them (compile-time rejection). Taking
`&self` keeps the estimator immutable and reusable, including concurrent fits.

*What we rejected:* Reusing one fit trait with an optional/ignored target parameter — that
would let callers pass meaningless targets and lose the type-level guarantee.

### `SKPredictor` (the prediction contract)

*What it is:* A trait with an associated `Error` whose `predict(&self, features)` returns
`Result<ndarray::Array1<f64>, Self::Error>` over the `TryInto` seam.

*Why it exists:* Prediction lives on the **fitted model**, never on the configured
estimator (spec `estimator-contracts`). That makes "predict before fit" a compile-time
error. `predict` returns a *dense* `Array1<f64>` — predicted scores or class indices.

*What we rejected:* Putting `predict` on the estimator type. That would break the "cannot
predict before fitting" guarantee and the shared model across threads.

### `SKFeatureTransformer` (the feature-transformer contract)

*What it is:* A trait with associated `Output` and `Error` types whose `transform(&self, features)`
returns `Result<Self::Output, Self::Error>` over the `TryInto` seam.

*Why it exists:* Transformers (e.g. scalers) need a typed `Output` so pipelines can check
at **compile time** that one stage's output feeds the next stage's input (spec
`estimator-contracts`). The associated type makes incompatible chaining fail early.

*What we rejected:* Returning an untyped or fixed output — that would push type
compatibility checking to runtime.

Here is the whole family at a glance:

```
                         sciencekit_common
                             |
      +-----------------------+----------------------+
      |                                              |
   numbers                                      data & errors
      |                                              |
   SKFloat (sealed) <-- f32, f64               SKError (enum)
      |                                              |
      +-- SKFloatSealed (private seal)         SKDataView (Dense | Sparse)
                                                SKTargetView (Continuous | Integer | Nominal)
                                                      |
                                     +----------------+----------------+
                                     |                                 |
                               SKSupervisedFit                     SKUnsupervisedFit
                               (features + targets)                (features only)
                                     |                                 |
                               SKPredictor (on the model)          (model type)
                               SKFeatureTransformer (transform)
```

---

## 5. A real walk-through, using the tests

The most convincing way to learn is to read actual tests that passed. Here are three real
tests from `data_view_tests.rs` and `sk_float_tests.rs`.

### Test 1: `borrowed_dense_enters_without_copying`

This test proves the zero-copy promise: a dense matrix enters the library without copying.

```rust
#[test]
fn borrowed_dense_enters_without_copying() {
    let data = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let view: SKDataView<'_, f64> = ArrayView2::from(&data).try_into().unwrap();
    match view {
        SKDataView::Dense(v) => {
            assert_eq!(v.as_slice_memory_order().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        }
        other => panic!("expected dense, got {other:?}"),
    }
}
```

Line by line:

1. `let data = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();` — we
   build an **owned** 2×2 dense matrix (2 rows, 2 columns) holding the four values
   `1.0, 2.0, 3.0, 4.0`. The `_f64` suffix says the numbers are 64-bit floats. `.unwrap()`
   unwraps the `Result`; here it cannot fail.
2. `let view: SKDataView<'_, f64> = ArrayView2::from(&data).try_into().unwrap();` — first we
   make a *view* of `data` (borrowing it, no copy), then we ask to convert it into a
   `SKDataView`. The `'_` lifetime says "this view is valid for as long as `data` lives."
   The conversion is the `TryFrom<ArrayView2>` impl we saw — it just wraps the view as
   `SKDataView::Dense`. No elements are copied.
3. `match view { ... }` — we check which variant we got. Since we handed it a dense view,
   we expect `SKDataView::Dense(v)`.
4. `assert_eq!(v.as_slice_memory_order().unwrap(), &[1.0, 2.0, 3.0, 4.0]);` — we confirm the
   dense view exposes exactly the four original values, in the same order. The `other =>
   panic!` arm catches the (impossible here) case where it came back sparse.

The lesson: the *same buffer* `data` is reused; the test's very name asserts no copy was
made.

### Test 2: `dense_only_consumer_rejects_sparse_precisely`

This test proves that an algorithm supporting only dense data rejects a sparse input with
a *precise* error, *before* touching any element.

```rust
#[test]
fn dense_only_consumer_rejects_sparse_precisely() {
    let csr = CsMat::new_csc((2, 2), vec![0, 0, 0], vec![], Vec::<f64>::new());
    let view: SKDataView<'_, f64> = csr.view().try_into().unwrap();
    let dense_result = view.as_dense();
    match dense_result {
        Err(SKError::UnsupportedRepresentation { representation, .. }) => {
            assert_eq!(representation, "csr");
        }
        other => panic!("expected unsupported-representation, got {other:?}"),
    }
}
```

Line by line:

1. `CsMat::new_csc((2, 2), vec![0, 0, 0], vec![], Vec::<f64>::new())` — build a 2×2 sparse
   matrix with no stored values (an empty sparse matrix). Note the `new_csc` here really
   produces a sparse matrix in the `sprs` crate's row-compressed form usable as CSR.
2. `let view: SKDataView<'_, f64> = csr.view().try_into().unwrap();` — take a *view* of the
   sparse matrix and convert it through the same seam. This time it becomes
   `SKDataView::Sparse(...)`.
3. `let dense_result = view.as_dense();` — we simulate a dense-only algorithm calling the
   dispatch method. Since the view is sparse, `as_dense()` returns an `Err`.
4. The `match` checks that the error is exactly `SKError::UnsupportedRepresentation` with
   `representation == "csr"` — not a vague shape error. This matches the spec: *"the
   operation fails before processing any element, with an error indicating the mismatch
   and the suggested conversion path."* The `other => panic!` arm catches any wrong error
   kind.

The lesson: representation errors are precise and distinct from shape errors, and they
happen up front — never after a wasted pass over the data.

### Test 3: `native_floats_satisfy_the_contract`

This test proves the sealed `SKFloat` works for both supported floats, through a generic
function.

```rust
#[test]
fn native_floats_satisfy_the_contract() {
    let x32 = scale(2.0_f32, 3.0_f32);
    assert_eq!(x32, 6.0_f32);

    let x64 = scale(2.0_f64, 3.0_f64);
    assert_eq!(x64, 6.0_f64);
}
```

The helper `scale` is defined just above it:

```rust
fn scale<F: SKFloat>(value: F, factor: F) -> F {
    value * factor
}
```

Line by line:

1. `fn scale<F: SKFloat>(value: F, factor: F) -> F` — a generic function over `F`, bounded
   by `SKFloat`. It takes two values and returns one, all of type `F`.
2. `value * factor` — the body uses `*`. This compiles only because `SKFloat: Float`
   guarantees multiplication exists. The function never needs to know whether `F` is
   `f32` or `f64`.
3. In the test, `scale(2.0_f32, 3.0_f32)` instantiates `F = f32` and yields `6.0_f32`;
   `scale(2.0_f64, 3.0_f64)` instantiates `F = f64` and yields `6.0_f64`. Both `assert_eq!`
   checks pass.

The companion test `integers_do_not_satisfy_the_contract` documents the other side: `i32`
does *not* implement `SKFloat`, so calling a continuous function with an integer is a
**compile-time** error. And `floats_are_thread_transferable` asserts `f32`/`f64` are
`Send + Sync + 'static`, guaranteeing they can cross thread boundaries — the basis for
future parallel algorithms.

---

## 6. Look inside

<div class="sk-box sk-box--tip">
  <p><strong>One strong insight:</strong> the entire <code>sciencekit_common</code>
  vocabulary is engineered around <em>type-level safety</em>. Sealing <code>SKFloat</code>,
  keeping <code>predict</code> off the estimator, making the fit traits reject missing
  targets at compile time, and using <code>TryInto</code> for an open conversion seam — all
  of these push mistakes out of the running program and into the compiler, where they are
  caught for free, before anything executes.</p>
</div>

Here is how the fit/predict/transform contracts relate to the estimator, the model, and
the data views:

```mermaid
flowchart TD
    subgraph est[Configured estimator — immutable, reusable]
        SF[SKSupervisedFit] -->|fit features + targets| M1(Model type)
        UF[SKUnsupervisedFit] -->|fit features only| M2(Model type)
    end

    subgraph model[Fitted model — sole bearer of learned state, thread-shareable]
        P[SKPredictor] -->|predict features| OUT[Array1 f64]
        FT[SKFeatureTransformer] -->|transform features| OUT2[Output type]
    end

    DV[SKDataView Dense|Sparse] --> SF
    DV --> UF
    DV --> P
    DV --> FT
    TV[SKTargetView Continuous|Integer|Nominal] --> SF

    ER[SKError] --> SF
    ER --> UF
    ER --> P
    ER --> FT

    style est fill:#e8f4f8,stroke:#6c8ebf
    style model fill:#e8f5e9,stroke:#7ea6a0
    style DV fill:#fff2cc,stroke:#d6b656
    style TV fill:#fff2cc,stroke:#d6b656
    style ER fill:#fce4ec,stroke:#c2185b
```

The diagram shows the two halves of the system: the estimator side (which *fits*) and the
model side (which *predicts/transforms*). The fitted model is deliberately a separate type
from the estimator, and both depend only on the shared data views and the central error.

---

## 7. Recap

- **`SKFloat` is sealed** — only `f32` and `f64` are valid continuous numbers, enforced at
  compile time by a private supertrait (`private::SKFloatSealed`).
- **`SKError` is one precise taxonomy** — shape, unsupported representation, hyperparameter,
  execution mode, batch overflow, convergence, I/O, and conversion — built with `thiserror`,
  `#[non_exhaustive]`, and automatic `From` conversion into per-algorithm errors.
- **Data enters through zero-copy views** — `SKDataView` (dense/sparse) and `SKTargetView`
  (continuous/integer/nominal) borrow the caller's buffers rather than copying them, which
  is the heart of the performance promise.
- **Public inputs are declared over `TryInto`** — any type that can convert into a view is
  accepted, making the library open to third-party integration without breaking.
- **Fit, predict, and transform are separate contracts** — `SKSupervisedFit` and
  `SKUnsupervisedFit` split by supervision; `SKPredictor` and `SKFeatureTransformer` live on
  the *model* type, making "predict before fit" a compile-time error.

*Next chapter:* now that we have the shared vocabulary of numbers, errors, and data views,
we can look at how an actual algorithm puts it to work — fitting a model and predicting on
new data using these very contracts.