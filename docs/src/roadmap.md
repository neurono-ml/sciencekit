# Plan & Roadmap

The roadmap follows the scikit-learn mapping faithfully, respecting dependencies between crates. Development methodology is as important as the destination:

<div class="sk-cards">
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--sky">📋</span>
    <div class="sk-card__title">OpenSpec-first</div>
    <p>Every change starts with a specification (proposal, deltas, design, tasks) on a dedicated <code>-openspec</code> branch, reviewed before any code.</p>
  </div>
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--violet">🔴</span>
    <div class="sk-card__title">Mandatory TDD</div>
    <p>Test first, confirmed failure, minimal implementation, refactoring — with companion <code>*_tests.rs</code> modules and mock data in <code>ndarray</code>/<code>sprs</code>.</p>
  </div>
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--emerald">🌳</span>
    <div class="sk-card__title">Worktree workflow</div>
    <p>No direct commits to <code>main</code>. Every change lives in a git worktree with small, focused PRs.</p>
  </div>
  <div class="sk-card">
    <span class="sk-card__icon sk-icon--amber">🔍</span>
    <div class="sk-card__title">Independent review</div>
    <p>An independent agent validates that the spec was effectively met before every PR.</p>
  </div>
</div>

## Iterative evolution per algorithm

No stage may be skipped — each algorithm walks the full path:

<div class="sk-bars reveal">
  <div class="sk-bars__row"><span class="sk-bars__label">1 · Naive implementation</span><div class="sk-bars__track"><div class="sk-bars__fill" style="width:100%;background:linear-gradient(90deg,#0ea5e9,#0284c7)"></div></div><span class="sk-bars__value">1st</span></div>
  <div class="sk-bars__row"><span class="sk-bars__label">2 · Unit tests (TDD)</span><div class="sk-bars__track"><div class="sk-bars__fill" style="width:100%;background:linear-gradient(90deg,#a78bfa,#7c3aed)"></div></div><span class="sk-bars__value">2nd</span></div>
  <div class="sk-bars__row"><span class="sk-bars__label">3 · Performance (SIMD/rayon/layout)</span><div class="sk-bars__track"><div class="sk-bars__fill" style="width:100%;background:linear-gradient(90deg,#fbbf24,#d97706)"></div></div><span class="sk-bars__value">3rd</span></div>
  <div class="sk-bars__row"><span class="sk-bars__label">4 · Streaming / out-of-core</span><div class="sk-bars__track"><div class="sk-bars__fill" style="width:100%;background:linear-gradient(90deg,#34d399,#059669)"></div></div><span class="sk-bars__value">4th</span></div>
</div>

## Acceptance checklist (every implementation)

- [ ] Runs correctly with lots of data and with little data
- [ ] Correct under concurrency (automatic + applicable explicit modes)
- [ ] Exports model (minimum Safetensors) and produces metrics
- [ ] Companion `_tests.rs` modules covering everything above
- [ ] Complete names, no abbreviations (single exception: `sk`/`SK`)
- [ ] Public items with mandatory prefix; methods unprefixed
- [ ] No file over 200 lines without folder-module organization
- [ ] Python binding in an associated change; GPU backends in separate changes

## Phases

<div class="sk-timeline">

<div class="sk-timeline__item is-active reveal">
<div class="sk-timeline__card">
<div class="sk-timeline__head"><span class="sk-timeline__title">Phase 0 — Foundations</span><span class="sk-pill sk-pill--amber">In progress</span></div>
<p style="margin:0.2rem 0 0.6rem;font-size:0.9rem;color:#52525c">Workspace, toolchain (Rust 1.85, edition 2024), CI, license · <code>sciencekit_common</code> traits + central errors · <code>sciencekit_math</code> kernel · automatic execution decision + builders + tracing/OTel.</p>
<div class="sk-progress"><div class="sk-progress__bar sk-progress__bar--amber" style="width:10%"></div></div>
<div class="sk-progress__meta"><span>2 of 20 roadmap items specced (bootstrap-workspace, common-core-foundation)</span><span>10%</span></div>
</div>
</div>

<div class="sk-timeline__item reveal">
<div class="sk-timeline__card">
<div class="sk-timeline__head"><span class="sk-timeline__title">Phase 1 — Preprocessing</span><span class="sk-pill sk-pill--neutral">Planned</span></div>
<p style="margin:0.2rem 0 0;font-size:0.9rem;color:#52525c"><code>SKStandardScaler</code>, <code>SKMinMaxScaler</code>, <code>SKRobustScaler</code>, <code>SKOneHotEncoder</code>, <code>SKPolynomialFeatures</code>; imputation strategies + <code>SKKNNImputer</code>.</p>
</div>
</div>

<div class="sk-timeline__item reveal">
<div class="sk-timeline__card">
<div class="sk-timeline__head"><span class="sk-timeline__title">Phase 2 — Linear models</span><span class="sk-pill sk-pill--neutral">Planned</span></div>
<p style="margin:0.2rem 0 0;font-size:0.9rem;color:#52525c">Linear/Ridge/Lasso/ElasticNet/Logistic regressions; SGD classifier/regressor with streaming (<code>SKLazySource</code>).</p>
</div>
</div>

<div class="sk-timeline__item reveal">
<div class="sk-timeline__card">
<div class="sk-timeline__head"><span class="sk-timeline__title">Phase 3 — Neighbors &amp; SVM</span><span class="sk-pill sk-pill--neutral">Planned</span></div>
<p style="margin:0.2rem 0 0;font-size:0.9rem;color:#52525c">k-NN classifier/regressor, KD-tree, Ball-tree; SVC, SVR, LinearSVC.</p>
</div>
</div>

<div class="sk-timeline__item reveal">
<div class="sk-timeline__card">
<div class="sk-timeline__head"><span class="sk-timeline__title">Phase 4 — Trees &amp; ensembles</span><span class="sk-pill sk-pill--neutral">Planned</span></div>
<p style="margin:0.2rem 0 0;font-size:0.9rem;color:#52525c">Decision trees; Random Forest, Bagging, Gradient Boosting aggregated in parallel with rayon.</p>
</div>
</div>

<div class="sk-timeline__item reveal">
<div class="sk-timeline__card">
<div class="sk-timeline__head"><span class="sk-timeline__title">Phase 5 — Unsupervised</span><span class="sk-pill sk-pill--neutral">Planned</span></div>
<p style="margin:0.2rem 0 0;font-size:0.9rem;color:#52525c">KMeans, MiniBatchKMeans (streaming), DBSCAN; PCA, TruncatedSVD; anomaly detection.</p>
</div>
</div>

<div class="sk-timeline__item reveal">
<div class="sk-timeline__card">
<div class="sk-timeline__head"><span class="sk-timeline__title">Phase 6 — Selection, pipelines, metrics</span><span class="sk-pill sk-pill--neutral">Planned</span></div>
<p style="margin:0.2rem 0 0;font-size:0.9rem;color:#52525c">KFold, train/test split, GridSearchCV; type-safe SKPipeline with DAGs; full metrics suite.</p>
</div>
</div>

<div class="sk-timeline__item reveal">
<div class="sk-timeline__card">
<div class="sk-timeline__head"><span class="sk-timeline__title">Phase 7 — Interop &amp; production</span><span class="sk-pill sk-pill--neutral">Planned</span></div>
<p style="margin:0.2rem 0 0;font-size:0.9rem;color:#52525c">Safetensors sharding/padding/compression, ONNX export/import, Polars/DataFusion sources; umbrella crate; complete PyO3 bindings.</p>
</div>
</div>

</div>

## Cross-cutting (continuous)

<div class="sk-cards">
  <div class="sk-box sk-box--info" style="margin-top:0">
  <strong>Per completed algorithm:</strong> Python binding in an associated change; GPU backend (OpenCL → CUDA → ROCm) in following separate changes.
  </div>
  <div class="sk-box sk-box--tip" style="margin-top:0">
  <strong>Recorded for the future:</strong> workflows CLI; ONNX/Safetensors import converted to native types with retraining/LoRA support.
  </div>
</div>

<div class="sk-btn-row">
  <a class="sk-btn sk-btn--star" href="https://github.com/neurono-ml/sciencekit/issues">Pick a task and help ⭐</a>
  <a class="sk-btn sk-btn--primary" href="./contribute.html">Contribution guide</a>
</div>
