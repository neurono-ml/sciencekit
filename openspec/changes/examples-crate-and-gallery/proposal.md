# Proposal

> Architecture Decision Record: #200

## Why

sciencekit has no way to *show* what it does. The scikit-learn gallery
(`https://scikit-learn.org/stable/auto_examples/index.html`, 34 sections, 281 example
pages) is the reference surface every ML practitioner learns from, and it is the natural
acceptance instrument for a library whose mission is to reimplement all of scikit-learn.
Without a runnable example corpus and a didactic book, the library's behavior — especially
the *effect of configuration choices* — can only be asserted in unit tests, never
demonstrated.

The examples must also become the contract for the multi-language bindings: the same
example, in the same configuration, expressed in Rust, Python, Go, Julia, JVM and Lua, is
what makes the bindings comparable and reviewable.

## What Changes

- **New workspace crate `sciencekit_examples`** (`crates/sciencekit_examples`), holding one
  cargo **example target** per example, runnable as `cargo example <example_id>`. The
  crate is **not published** to crates.io, and its whole surface is behind a
  **non-default `examples` feature**, so it is neither compiled nor built when the
  library is used.
- **Folder-module layout per example family and per example**, following the repository
  convention (`mod.rs` is a pure dispatcher; every file stays under 200 lines; tests live in
  companion `*_tests.rs` modules). No example logic lives in `main.rs`.
- **A local dataset module owned by the crate** (`sciencekit_examples::datasets`),
  reproducing the `sklearn.datasets` surface the gallery depends on: the `make_*` synthetic
  generators and the small classic datasets, with no network access at run time.
- **A design record per example** (the FDR set): the dataset and its origin, the
  feature selection, the feature engineering, the preprocessing, the ordered pipeline,
  the configuration spectrum, the behavior the reader must observe, the reported output
  and the sciencekit capability it needs — going further than the scikit-learn original
  on exactly the points the corpus exists to teach.
- **A machine-readable example catalog** recording, for every scikit-learn gallery entry:
  the example id, its family, the exact scikit-learn source file it is ported from, the
  dataset source, the sciencekit estimators involved, the configuration knobs exercised,
  the supported languages and the porting status — including the entries intentionally not
  ported and the reason.
- **A porting guide** (code style and per-example instructions) so that the agent or
  contributor porting an example knows the scikit-learn source to read, the dataset to
  reuse, the estimator to adapt and the naming, layout and documentation rules to follow.
- **An "Examples" documentation section** in the main mdBook: a gallery organized by
  algorithm family, mirroring the scikit-learn gallery's taxonomy, one page per example.
- **A separate "Machine Learning with Sciencekit" book**, referenced from the main book,
  teaching the algorithms as machine learning rather than as API calls: every chapter
  explains the concept from zero, shows the configuration spectrum of the algorithm and
  makes the cause-and-effect of each knob explicit, grounded in the real example code and
  the real `*_tests.rs` modules.
- **A plotting abstraction** so that every example can produce its figure without
  blocking the crate on an unready dependency: the plotting backend is an internal trait
  with a `scienceview`-backed implementation behind a non-default Cargo feature, which
  stays disabled until `scienceview` publishes a usable API.
- **Deferred (future change, not in this one):** the multi-language tab rendering of every
  example. It is recorded here so it is not lost, and it is sequenced after the bindings
  API implementation and after this crate's code exists.

## Capabilities

### New Capabilities

- `examples-crate`: the `sciencekit_examples` crate — workspace integration, per-family and
  per-example folder modules, thin cargo example targets named `<example_id>`, the naming rules without
  abbreviations, and the internal plotting abstraction with its pluggable backend.
- `example-datasets`: the dataset surface owned by the examples crate — synthetic `make_*`
  generators, the small classic datasets, deterministic seeding and the no-network rule.
- `example-catalog`: the machine-readable catalog of the scikit-learn gallery parity —
  one record per gallery entry with its scikit-learn source, dataset source, algorithms,
  configuration knobs, languages, status and the reason for every intentional omission.
- `examples-documentation`: the "Examples" gallery section of the main book and the
  separate "Machine Learning with Sciencekit" book — page structure, taxonomy, the
  didactic contract of the book and the cross-references between both.
- `example-design-records`: the FDR set — the design of every example, its pipeline
  recipe, its configuration spectrum and the capability it needs; the catalog is derived
  from it.

### Modified Capabilities

- `docs-book-theme`: the main book's summary gains the "Examples" section and the
  reference to the separate book; the book theme gains the example page and gallery
  navigation structures.

## Impact

- **Workspace**: `Cargo.toml` gains `crates/sciencekit_examples` as a member; the crate
  declares `publish = false`, so nothing of it reaches crates.io. The root umbrella crate
  is untouched.
- **New code**: `crates/sciencekit_examples/**` — crate scaffold, dataset module, catalog,
  plotting abstraction, the per-example logic, and the cargo example targets.
- **Tooling**: `.cargo/config.toml` gains the `example` alias, so the corpus is run with
  `cargo example <example_id>` from the repository root.
- **Documentation**: `docs/src/**` for the "Examples" section and the gallery pages; a new
  separate book for "Machine Learning with Sciencekit"; `docs/skin/**` for the gallery and
  book navigation styles.
- **Dependencies**: no new runtime dependency. `scienceview` is referenced only as an
  optional, non-default Cargo feature and is not published to crates.io yet; `sciencekit`
  itself is the only mandatory dependency.
- **Sequencing**: the catalog records, per example, the sciencekit crate it depends on, so
  the corpus grows as the algorithms land, family by family.
- **Not affected**: the published `sciencekit` crate — no example source, no example
  feature and no example target reaches crates.io.
