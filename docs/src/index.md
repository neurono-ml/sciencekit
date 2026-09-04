# sciencekit

<div class="sk-hero reveal">
  <div class="sk-announce">
    <span class="sk-announce__tag">New</span>
    In active development · Apache-2.0
  </div>
  <h1 class="sk-hero__title">All of scikit-learn. Natively in Rust.</h1>
  <p class="sk-hero__subtitle">
    <strong>sciencekit</strong> reimplements every algorithm and utility of
    scikit-learn from scratch, with extreme performance, memory safety,
    zero-copy public APIs and native support for datasets larger than RAM.
  </p>
  <div class="sk-btn-row" style="justify-content: center;">
    <a class="sk-btn sk-btn--star" href="https://github.com/neurono-ml/sciencekit/stargazers">⭐ Star on GitHub</a>
    <a class="sk-btn sk-btn--primary" href="./algorithms.html">Browse the algorithms</a>
    <a class="sk-btn sk-btn--secondary" href="https://github.com/neurono-ml/sciencekit">GitHub ↗</a>
  </div>
</div>

<div class="sk-stats reveal">
  <div class="sk-stats__item"><div class="sk-stats__value js-count" data-target="39" data-suffix="+">39+</div><span class="sk-stats__label">algorithms mapped</span></div>
  <div class="sk-stats__item"><div class="sk-stats__value js-count" data-target="19">19</div><span class="sk-stats__label">crates in the workspace</span></div>
  <div class="sk-stats__item"><div class="sk-stats__value js-count" data-target="8">8</div><span class="sk-stats__label">roadmap phases</span></div>
  <div class="sk-stats__item"><div class="sk-stats__value">100%</div><span class="sk-stats__label">pure Rust by default</span></div>
</div>

## Why does sciencekit exist?

Python ML libraries carry decades of ecosystem — along with the GIL, insecure pickle,
and layers of interpretation between your code and the metal. sciencekit brings the same
algorithmic reach as scikit-learn to a compiled, native base — no garbage collector,
no ergonomic compromises:

<div class="sk-cards reveal">
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--sky">⚡</span>
    <div class="sk-card__title">Extreme performance</div>
    <p>Native code, SIMD, data parallelism with rayon, and custom allocators selected automatically for your workload.</p>
  </div>
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--violet">🛡</span>
    <div class="sk-card__title">Memory safety</div>
    <p>All of Rust's guarantees without a garbage collector. Goodbye pickle: models travel as auditable Safetensors or ONNX.</p>
  </div>
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--emerald">💾</span>
    <div class="sk-card__title">Native out-of-core</div>
    <p>Streaming batches or memory-mapping across every algorithm — datasets larger than RAM stop being a blocker.</p>
  </div>
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--amber">🧭</span>
    <div class="sk-card__title">First-class concurrency</div>
    <p>You choose the execution mode — or the automatic decision engine does: synchronous, asynchronous, streaming or memmap.</p>
  </div>
</div>

## The API you will write

Mandatory builder pattern, zero-copy inputs, and full descriptive names — no `max_iter` anywhere:

```rust,ignore
// Illustrative — requires the sciencekit crates (Phase 0+).
let model = SKKMeansClassifierBuilder::new()
    .number_of_clusters(8)
    .maximum_iterations(300)
    .execution_mode(SKExecutionMode::Automatic)
    .build()?;

model.fit(&training_data_view)?;
let predictions = model.predict(&test_data_view)?;
```

Inputs take `ArrayView`/`CowArray` (or sparse views via `sprs`) — never copies of huge matrices.
Pipelines are validated at compile time.

## Try it in your browser

The example below mirrors the exact shape of the sciencekit API — mandatory builder,
full descriptive names, `execution_mode` on every builder. Edit the code and hit **Run**:
it executes for real, right on the page.

```rust,editable
// The sciencekit API follows the builder pattern everywhere.
// This standalone demo mirrors its shape — the real crates arrive with Phase 0.
// Bounded options are enums, never strings, so `execution_mode` cannot be
// misspelled.
#[derive(Debug, Clone, Copy)]
enum SKExecutionMode {
    Automatic,
    InProcessSynchronous,
    OutOfCoreStreaming,
}

#[derive(Debug)]
struct TrainedKMeansModel {
    number_of_clusters: usize,
    maximum_iterations: usize,
    execution_mode: SKExecutionMode,
}

struct KMeansBuilder {
    number_of_clusters: usize,
    maximum_iterations: usize,
    execution_mode: SKExecutionMode,
}

impl KMeansBuilder {
    fn new() -> Self {
        Self {
            number_of_clusters: 3,
            maximum_iterations: 100,
            execution_mode: SKExecutionMode::Automatic,
        }
    }

    fn number_of_clusters(mut self, value: usize) -> Self {
        self.number_of_clusters = value;
        self
    }

    fn maximum_iterations(mut self, value: usize) -> Self {
        self.maximum_iterations = value;
        self
    }

    fn execution_mode(self, mode: SKExecutionMode) -> Self {
        println!("execution mode requested: {mode:?}");
        self
    }

    fn build(self) -> TrainedKMeansModel {
        TrainedKMeansModel {
            number_of_clusters: self.number_of_clusters,
            maximum_iterations: self.maximum_iterations,
            execution_mode: self.execution_mode,
        }
    }
}

fn main() {
    let model = KMeansBuilder::new()
        .number_of_clusters(8)
        .maximum_iterations(300)
        .execution_mode(SKExecutionMode::Automatic)
        .build();

    println!("model ready: {model:?}");
}
```

<div class="sk-box sk-box--tip">
<strong>How this works:</strong> the <strong>Run</strong> button compiles and executes the code on the
official Rust Playground. Once the real <code>sciencekit</code> crates are published, the same examples
will run against the library itself.
</div>

<div class="sk-box sk-box--info">
<strong>Current status:</strong> Phase 0 (foundations) is complete — Cargo workspace, core traits
(<code>SKEstimator</code>, <code>SKPredictor</code>, <code>SKTransformer</code>), the pure-Rust <code>faer</code> math
kernel, and the automatic execution-decision + observability layer have all landed. Next up is
Phase 1 (preprocessing). Follow the full plan in <a href="./roadmap.html">Plan &amp; Roadmap</a>.
</div>

## Who is it for?

- **ML engineers in Rust** who need a complete, idiomatic toolkit.
- **Python teams** that want high-performance inference/training through PyO3 bindings with maximum zero-copy.
- **Production systems** where scikit-learn's pickle is a security risk and every millisecond counts.

<div class="sk-statement reveal">
  <p class="sk-statement__text">The complete reach of scikit-learn,<br/>with the speed and safety of Rust.</p>
</div>

<div class="sk-cta reveal">
  <div class="sk-cta__title">Build it with us</div>
  <p>The project is open source (Apache-2.0) and advances algorithm by algorithm — naive implementation,
  tests, performance, out-of-core. Every kind of help is welcome, from docs to GPU backends.</p>
  <div class="sk-btn-row" style="justify-content: center;">
    <a class="sk-btn sk-btn--star" href="https://github.com/neurono-ml/sciencekit/stargazers">⭐ Leave a star</a>
    <a class="sk-btn sk-btn--primary" href="./contribute.html">How to contribute</a>
  </div>
</div>
