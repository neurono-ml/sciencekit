# Tasks

## 1. Workspace and crate scaffold

- [ ] 1.1 Add `crates/sciencekit_examples` to the workspace members and create the crate manifest with `sciencekit` as its only mandatory dependency and `publish = false`, verifying `cargo build -p sciencekit_examples` succeeds offline and `cargo package -p sciencekit_examples` refuses to publish
- [ ] 1.2 Create the crate skeleton (`src/lib.rs`, `src/examples/mod.rs`, `src/datasets/mod.rs`, `src/catalog/mod.rs`, `src/plotting/mod.rs`) with pure-dispatcher `mod.rs` files behind the non-default `examples` feature, verifying each file stays under 200 lines and contains no implementation logic
- [ ] 1.3 Declare the `examples` feature as non-default and gate every module declaration behind it, verifying `cargo test --workspace` compiles nothing of the corpus and `cargo test -p sciencekit_examples --features examples` compiles and runs it
- [ ] 1.4 Add the `example` alias to `.cargo/config.toml` (`run -p sciencekit_examples --features examples --example`) and verify `cargo example <id>` runs a target, verifying the alias resolves from the repository root
- [ ] 1.5 Declare the first example target with `required-features = ["examples"]` and verify that a deliberately broken example target does not break `cargo test --workspace`, verifying the workspace test stays green while the target is broken
- [ ] 1.6 Implement the manifest block generator that rewrites the `[[example]]` section between marker comments from the examples directory, verifying the generated block declares every example with `required-features = ["examples"]`
- [ ] 1.7 Add the test asserting the committed manifest block matches the examples directory, verifying the test fails when an example directory is added without regenerating the block

## 2. Dataset surface

- [ ] 2.1 Implement the synthetic generators (`make_classification`, `make_blobs`, `make_circles`, `make_moons`, `make_regression`, `make_swiss_roll`) with explicit seeds and scikit-learn parameter semantics, verifying the companion tests assert the documented shapes and that the same seed yields equal designs
- [ ] 2.2 Implement the sparse-capable generators (`make_sparse_spd_matrix`, `make_sparse_uncorrelated`, `make_low_rank_matrix`, `make_multilabel_classification`), verifying the returned design stays a sparse view and is never densified
- [ ] 2.3 Implement the missing-value introduction used by the imputation examples, verifying the introduced fraction matches the request and the mechanism is recorded
- [ ] 2.4 Embed the classic datasets (iris, digits, wine, diabetes, breast cancer, linnerud) with their real shapes, feature names and target names, verifying the companion tests assert each shape and that loading performs no network access
- [ ] 2.5 Ship a provenance record per embedded dataset (original source, license, citation, transformations applied), verifying the dataset module's test fails when a dataset has no provenance record

## 3. Catalog

- [ ] 3.1 Define the catalog record schema and its loader, with the fields `id`, `family`, `title`, `sklearn_source`, `sklearn_url`, `dataset_source`, `sciencekit_crates`, `estimators`, `knobs`, `languages`, `status`, `blocked_by`, `omission_reason`, verifying the loader rejects a record missing a required field
- [ ] 3.2 Populate the catalog with one record for each of the 281 scikit-learn gallery entries, recording every entry's family, scikit-learn source path and URL, and dataset source, and verifying the count against the gallery index
- [ ] 3.3 Mark the entries that are intentionally not ported (release highlights, developer-facing entries, text-corpus entries) with `status: omitted` and a recorded reason, verifying no entry is missing from the catalog
- [ ] 3.4 Mark the entries blocked by a sciencekit capability that does not exist yet with `status: blocked` and the missing capability in `blocked_by`, verifying every blocked record names its blocker
- [ ] 3.5 Add the catalog validation test that cross-checks the catalog against the crate's filesystem (every ported record has a module and an example target, every module has a record), verifying the test fails when a record is added without its module

## 4. Plotting abstraction

- [ ] 4.1 Define the internal plotting surface (line series, scatter, bar, histogram, image grid, contour) as a trait plus a data-oriented implementation, verifying the companion tests assert the emitted primitives
- [ ] 4.2 Implement the reporting helpers used by the examples to print metrics and tables in a single format, verifying the helpers render the same numbers the gallery page quotes
- [ ] 4.3 Isolate the `scienceview`-backed implementation in `src/plotting/scienceview_backend.rs` behind the non-default `plots` feature, verifying no other module in the crate depends on a plotting library
- [ ] 4.4 Add the per-example cargo target convention (`examples/<id>/main.rs` as a shim over the library entry point) to the porting guide, verifying a new example needs exactly one `[[example]]` manifest entry and one target directory

## 5. Porting guide and code style

- [ ] 5.1 Write the porting guide (the scikit-learn source layout, the seven-step porting recipe, the estimator mapping table, the allowed mathematical tokens, the code-style rules) under `docs/src/`, verifying every step names the artifact it produces
- [ ] 5.2 Add the catalog's companion document that maps each scikit-learn gallery section to its sciencekit crate, its status and its wave, verifying it is generated from the catalog
- [ ] 5.3 Add a lint-style check that the crate contains no abbreviated identifiers other than the allowed mathematical tokens, verifying the check reports a deliberately introduced abbreviation

## 6. First wave — linear models

- [ ] 6.1 Port `ordinary_least_squares_and_ridge_regression` (`examples/linear_model/plot_ols_ridge.py`): dataset from the embedded diabetes dataset, `SKLinearRegression` and `SKRidge` with the `alpha ∈ {0.0, 1.0, 10.0, 100.0}` spectrum, verifying the companion test asserts the quoted metrics
- [ ] 6.2 Port `lasso_and_elastic_net_on_sparse_signals` (`examples/linear_model/plot_lasso_and_elasticnet.py`) with the sparsity spectrum, verifying the companion test asserts the support behavior
- [ ] 6.3 Port `regularization_path_of_l1_logistic_regression` (`examples/linear_model/plot_logistic_path.py`), verifying the companion test asserts the path behavior
- [ ] 6.4 Port `ridge_coefficients_as_a_function_of_the_l2_regularization` (`examples/linear_model/plot_ridge_coeffs.py`), verifying the companion test asserts the coefficient behavior
- [ ] 6.5 Port `penalized_logistic_regression_convergence` (`examples/linear_model/plot_sgd_early_stopping.py` family) as a SGD configuration spectrum, verifying the companion test asserts the convergence behavior
- [ ] 6.6 Register the ported linear-model records in the catalog with their knob spectra and dataset sources, verifying the catalog validation test passes

## 7. First wave — preprocessing, imputation and model selection

- [ ] 7.1 Port the scaler comparison entry from the preprocessing family with the scaler spectrum, verifying the companion test asserts the scaling behavior
- [ ] 7.2 Port the polynomial-features entry from the preprocessing family with the degree spectrum, verifying the companion test asserts the expanded design
- [ ] 7.3 Port `imputing_missing_values_before_building_an_estimator` (`examples/impute/plot_missing_values.py`) using the missing-value generators, verifying the companion test asserts the imputation effect
- [ ] 7.4 Port the metric-evaluation entries from the model-selection family using the metrics crate, verifying the companion tests assert the reported scores
- [ ] 7.5 Register the ported records in the catalog with their knob spectra, verifying the catalog validation test passes

## 8. Pipelines

- [ ] 8.1 Port the pipeline construction entries (`examples/compose/*`) once the pipeline crate lands, using the catalog's `blocked_by` entries, verifying the companion tests assert the pipeline's end-to-end metrics
- [ ] 8.2 Port the pipeline display entry with the configuration spectrum it compares, verifying the companion test asserts the rendered pipeline description

## 9. Documentation — gallery

- [ ] 9.1 Implement the language-tab widget in the book's skin (grouped code blocks become one tab group, keyboard navigable, degrades to plain blocks without JavaScript), verifying the built book shows working tabs
- [ ] 9.2 Add the "Examples" section to the main book's summary with one family page per ported family, verifying `mdbook build docs` succeeds and the family pages appear in the sidebar
- [ ] 9.3 Implement the gallery page generator that produces one page per ported record, inlining the example's source files as the code blocks and taking the prose from the catalog record and the companion test, verifying a generated page's code block matches the source file byte for byte
- [ ] 9.4 Implement the gallery index generation from the catalog, verifying the index and the catalog agree
- [ ] 9.5 Add the gallery navigation styling to the book theme (family grouping, active page, cross-book link styling), verifying the sidebar exposes the family pages as a collapsible section
- [ ] 9.6 Publish the ported example pages of the first wave, verifying each page states the dataset origin, the knob spectrum, the run command and the expected output, carries the scikit-learn credit line, and links to its chapter
- [ ] 9.7 Document the rounding precision and the tolerance the quoted numbers use, verifying the companion test asserts within the documented tolerance and two platforms agree
- [ ] 9.8 Implement the gallery generator as a tool binary of the examples crate (`--bin gallery-generator`), verifying it writes the main book's gallery pages and the machine-learning book's pages from the corpus
- [ ] 9.9 Add the staleness check to the documentation workflow: regenerate the pages from the pinned corpus revision and fail when the result differs from the committed pages, verifying an intentionally outdated page fails the documentation build

## 10. Documentation — the separate machine-learning book

- [ ] 10.1 Create the separate book as a sibling of `docs/` on the `docs/documentations` branch (own `book.toml`, own summary, own theme, own Pages deployment workflow), verifying it builds independently of the main book
- [ ] 10.2 Record the `Development Explained` exemption for the example corpus in `AGENTS.md` (the machine-learning book carries the didactic duty for this area), verifying the rule states the exemption and its reason
- [ ] 10.3 Write the first chapter — regularization and the bias–variance tradeoff — grounded in the ported ridge example and its companion test, verifying every claim in the chapter is asserted by a test
- [ ] 10.4 Write the scaling chapter (why and when to standardize) grounded in the ported scaler example, verifying every claim in the chapter is asserted by a test
- [ ] 10.5 Write the imputation chapter grounded in the ported imputation example, verifying every claim in the chapter is asserted by a test
- [ ] 10.6 Add the cross-references between the gallery pages and the chapters in both directions, verifying no page or chapter lacks its counterpart link
- [ ] 10.7 Reference the separate book from the main book's summary, verifying the link renders, states the target book and is distinguishable from an ordinary internal link

## 11. Integration checks

- [ ] 11.1 Run the repository gates — `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`, `cargo test --workspace` — and verify all three are green
- [ ] 11.1a Run the corpus gate — `cargo test -p sciencekit_examples --features examples` — and verify the companion tests of every ported example are green
- [ ] 11.1b Record the corpus gate in `AGENTS.md` next to the repository gates, verifying the documented gate list names the command and its feature
- [ ] 11.2 Verify every ported example target runs and prints the metrics quoted on its gallery page, by running each target in the corpus of the first wave
- [ ] 11.3 Verify the internal parity check lists every gallery entry exactly once with its status and its blocker, and that the count of ported entries equals the count of example pages
- [ ] 11.4 Verify the generator is deterministic — running it twice on an unchanged corpus produces byte-identical pages — and that the committed pages match a fresh generation

## Workflow follow-up

- Deferred change — **multi-language tabs for the gallery**: create it after the bindings
  API implementation lands and after this change's code exists. It adds the Python, Go,
  Julia, JVM and Lua variants of every ported example, with the same example id, dataset
  and knob spectrum, and the idiomatic plotting library of each language
  (`matplotlib`, `gonum/plot`, `Plots`/`Makie`, and the closest supported option for JVM
  and Lua). The catalog's `languages` field and the tab widget are already in place for it.
- Deferred change — **`scienceview` plotting backend**: create it when `scienceview`
  publishes a usable data API; it fills `src/plotting/scienceview_backend.rs` and enables
  the `plots` feature by default.
- Deferred change — **promotion of the dataset surface to a `sciencekit_datasets`
  crate**: create it when the dataset surface needs to be public API for consumers beyond
  the examples.
- Archive the change after the project's review requirements are satisfied.
- Verify the archived result.
