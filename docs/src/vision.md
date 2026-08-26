# Project Vision

<div class="sk-announce">
  <span class="sk-announce__tag">The idea</span>
  A complete, fast and safe ML library — keeping everything that makes scikit-learn loved.
</div>

## Goal

`sciencekit` is a Machine Learning library written natively in Rust exposing
**every algorithm and utility of scikit-learn**, rewritten from scratch in an optimized form.
It is not a rushed port: each algorithm evolves through mandatory stages — naive implementation,
tests, performance tuning (SIMD, parallelism, memory layout) and out-of-core support.

<div class="sk-cards sk-cards--wide reveal">
  <div class="sk-card sk-card--flow">
    <span class="sk-card__icon sk-icon--sky">🚀</span>
    <div class="sk-card__title">Extreme performance</div>
    <p>Native code with autovectorization, explicit SIMD on profiled hot paths, data parallelism via
    <code>rayon</code>, and custom allocators (jemalloc/mimalloc) with an expected 15–30% gain in
    allocation throughput.</p>
  </div>
  <div class="sk-card sk-card--flow">
    <span class="sk-card__icon sk-icon--violet">🔐</span>
    <div class="sk-card__title">Memory safety</div>
    <p>All of Rust's guarantees without a garbage collector. Models travel as Safetensors — an auditable
    format that eliminates pickle's attack surface.</p>
  </div>
  <div class="sk-card sk-card--flow">
    <span class="sk-card__icon sk-icon--emerald">🌊</span>
    <div class="sk-card__title">Native out-of-core</div>
    <p>Two complementary traits cover any dataset: <code>SKLazySource</code> for sequential streaming
    in batches and <code>SKMappableSource</code> for O(1) random access via memory-mapping.</p>
  </div>
  <div class="sk-card sk-card--flow">
    <span class="sk-card__icon sk-icon--amber">⚙️</span>
    <div class="sk-card__title">First-class concurrency</div>
    <p>Computation never blocks async threads: rayon runs the math, Tokio handles I/O. The execution mode
    is user-configurable — or decided automatically from available RAM, dataset size and access pattern.</p>
  </div>
</div>

## Positioning

Equivalent to scikit-learn in algorithmic coverage, with the performance and safety of Rust.
Three target audiences:

| Audience | Need served |
|---|---|
| ML engineers in Rust | A complete, idiomatic toolkit without rewriting algorithms |
| Python teams | High-performance inference/training via zero-copy PyO3 bindings |
| Production systems | Removing the pickle risk + models exportable to ONNX/Safetensors |

## Scope

<div class="sk-cards">
  <div class="sk-box sk-box--success" style="margin-top:0">
  <strong>Is:</strong> a Rust library in a workspace with sub-crates; an idiomatic native API with the
  builder pattern; Python bindings (PyO3) with zero-copy; complete coverage of the scikit-learn taxonomy;
  ONNX, Safetensors and JSON debug export/import.
  </div>
  <div class="sk-box sk-box--danger" style="margin-top:0">
  <strong>Is not (for now):</strong> a CLI for running workflows (recorded as a future improvement);
  a deep learning framework; an inference server.
  </div>
</div>

## API principles

1. **Mandatory builder pattern** — every estimator, transformer and pipeline exposes a builder;
   direct constructors stay private. Every builder accepts `execution_mode(SKExecutionMode::...)`,
   defaulting to `Automatic`.
2. **Zero-copy inputs** — public APIs receive `ArrayView`, `CowArray` or sparse views (`sprs`),
   never arrays by value.
3. **Type-safe pipelines** — step compatibility validated at compile time through associated types.
4. **Full descriptive names** — no abbreviations; the single exception is the project prefix `sk`/`SK`.
   Examples: `maximum_number_of_iterations` (not `max_iter`), `nearest_neighbors_count` (not `k`).

```rust,ignore
// Core trait vocabulary — implemented by the sciencekit crates.
pub trait SKEstimator { /* hyperparameters and fit */ }
pub trait SKPredictor: SKEstimator { /* predict */ }
pub trait SKTransformer: SKEstimator { /* transform / fit_transform */ }

pub trait SKDataSource { /* eager: full in-memory access */ }
pub trait SKLazySource { /* streaming: Iterator<Item = Batch> */ }
pub trait SKMappableSource { /* memmap: O(1) random access */ }
```

Algorithms compose exactly the traits they support: an `SKSGDClassifier` implements
`SKPredictor` + `SKLazySource`; an `SKStandardScaler` implements only `SKTransformer`.

## Interoperability

- **Single internal format:** extended Safetensors — JSON header with hyperparameters, training state
  and recoverability metadata (checkpointing), with partial writes via sharding or header padding.
- **ONNX:** every estimator implements `SKToOnnx`; external models load as a generic `SKPredictor`.
- **Data:** pluggable sources behind arbitrary I/O traits, with conversions for Polars and DataFusion.
- **Observability:** `tracing` oriented to OpenTelemetry, disableable at near-zero cost.

<div class="sk-box sk-box--tip">
<strong>Source of truth:</strong> this site mirrors the PRD (<code>docs/PRD.md</code>). Relevant scope
changes go through that document first, before becoming technical specs.
</div>

<div class="sk-btn-row">
  <a class="sk-btn sk-btn--primary" href="./algorithms.html">Next: the algorithms →</a>
  <a class="sk-btn sk-btn--secondary" href="./architecture.html">Architecture</a>
</div>
