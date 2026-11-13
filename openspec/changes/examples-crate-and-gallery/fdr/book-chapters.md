# Book — the design records of the chapters

The chapter design records. A chapter record states the chapter's purpose, its
sections, the examples it uses, the figures it carries, the metadata it needs and the
two platform variants it produces.

The record schema is the one of `README.md`, adapted: a chapter is not an algorithm, so
the *Spectrum* stage becomes the **configuration discussion** and the *Effect* stage
becomes **what the reader must be able to do afterwards**.

---

### `what-is-sciencekit`

| Field | Value |
|---|---|
| Part | Opening |
| Purpose | Say what the library is, why it is written in Rust, and what the ecosystem already gives — the Python comparison, fairly stated |
| Sections | the library in one page · why Rust · what Rust gives that Python does not · what Python has and Rust does not · the algorithms that exist today · the capabilities: metrics, artifact export, observability, interop · where to start |
| Examples | none — the chapter argues, it does not run |
| Figures | the capability map (Mermaid) · the crate map (Mermaid) |
| Metadata | description: what sciencekit is and who it is for; keywords: machine learning in Rust, scikit-learn alternative, Rust ML library |
| Medium | "Machine learning in Rust, from scratch: what sciencekit is and why it exists" — opens with the friction of the Python stack, not with the project |
| LinkedIn | "Why I am reimplementing scikit-learn in Rust" — the professional stakes: deployment, memory safety, no GIL |

---

### `a-complete-example`

| Field | Value |
|---|---|
| Part | The complete example |
| Purpose | Take one simple algorithm from raw arrays to a saved model and a command-line inference, touching every capability of the library exactly once |
| Sections | the algorithm's focus · the data as arrays · the same data from polars · the same data lazily from datafusion · the training run and its exported metrics · the configuration comparison · the pipeline with preprocessing · the artifact: ONNX and safetensors, with netron diagrams · loading it back · the command-line inference over a dataset · how the data was split: holdout, cross-validation, segmentation |
| Examples | `ordinary_least_squares_and_ridge_regression` · `streaming_scaler_under_a_shifting_distribution` · `scaling_inside_a_cross_validated_pipeline` |
| Figures | the end-to-end flow (Mermaid) · the ONNX and safetensors layouts side by side (Mermaid) |
| Metadata | description: the end-to-end sciencekit walkthrough; keywords: sciencekit tutorial, ONNX export, polars, datafusion lazy loading, OpenTelemetry metrics |
| Medium | "From arrays to a served model in Rust: a complete sciencekit walkthrough" |
| LinkedIn | "The whole ML lifecycle in one Rust program: data, training, export, inference" |

---

### `preprocessing-and-pipelines`

| Field | Value |
|---|---|
| Part | Algorithms |
| Purpose | Teach the vocabulary every later algorithm reuses: transformations, feature engineering and the pipeline |
| Sections | the pipeline as a first-class object · feature engineering · feature selection · then one level-four subsection per algorithm: `SKStandardScaler`, `SKMinMaxScaler`, `SKRobustScaler`, `SKOneHotEncoder`, `SKPolynomialFeatures`, `SKSimpleImputer`, `SKMissingIndicator`, `SKKNNImputer`, `SKPipeline`, `SKFeatureUnion`, `SKColumnSelection` |
| Examples | `standard_scaler_on_a_mixed_scale_dataset` · `minmax_scaler_to_a_bounded_range` · `robust_scaler_against_outliers` · `one_hot_encoding_of_categorical_features` · `polynomial_feature_expansion_of_interactions` · `imputing_missing_values_before_building_an_estimator` · `missing_indicator_as_an_explicit_signal` · `knn_imputation_of_missing_feature_values` · `scaling_inside_a_cross_validated_pipeline` · `preprocessing_chain_cheat_sheet` |
| Figures | the pipeline as a graph (Mermaid) |
| Metadata | description: the preprocessing and pipeline vocabulary of sciencekit; keywords: feature engineering, feature selection, preprocessing pipeline, Rust ML |
| Medium | "Feature engineering is the model: building preprocessing pipelines in Rust" |
| LinkedIn | "Most of the win in ML is before the model: the sciencekit pipeline vocabulary" |

---

### `supervised-learning`

| Field | Value |
|---|---|
| Part | Algorithms |
| Purpose | Every supervised estimator, with a complete example each |
| Sections | classification · regression · then one level-four subsection per estimator: `SKLogisticRegression`, `SKLinearRegression`, `SKRidge`, `SKLasso`, `SKElasticNet`, `SKSGDClassifier`, `SKSGDRegressor`, `SKDecisionTreeClassifier`, `SKDecisionTreeRegressor`, `SKRandomForestClassifier`, `SKRandomForestRegressor`, `SKBaggingClassifier`, `SKBaggingRegressor`, `SKGradientBoostingClassifier`, `SKGradientBoostingRegressor`, `SKHistGradientBoostingClassifier`, `SKHistGradientBoostingRegressor`, `SKKNeighborsClassifier`, `SKKNeighborsRegressor`, `SKSVC`, `SKSVR`, `SKLinearSVC` |
| Examples | the supervised records of `02-linear-models.md`, `03-neighbors-and-svm.md` and `04-trees-and-ensembles.md` |
| Figures | the estimator map (Mermaid) |
| Metadata | description: the supervised estimators of sciencekit, each with a complete example; keywords: supervised learning, classification, regression, scikit-learn in Rust |
| Medium | "Twenty-two supervised estimators in Rust, each with the example that shows what it is for" |
| LinkedIn | "Choosing between ridge, lasso, boosting and SVMs — the same estimators, the same knobs, in Rust" |

---

### `unsupervised-learning`

| Field | Value |
|---|---|
| Part | Algorithms |
| Purpose | Every unsupervised estimator, with a complete example each |
| Sections | clustering · dimensionality reduction · then one level-four subsection per estimator: `SKKMeans`, `SKMiniBatchKMeans`, `SKDBSCAN`, `SKPCA`, `SKIncrementalPCA`, `SKTruncatedSVD` |
| Examples | the clustering records of `05-clustering.md` and the decomposition records of `06-decomposition.md` |
| Figures | the geometry of the estimators (Mermaid) |
| Metadata | description: the unsupervised estimators of sciencekit; keywords: clustering, k-means, DBSCAN, PCA, SVD, unsupervised learning |
| Medium | "Clustering and decomposition in Rust: when k-means is the wrong answer" |
| LinkedIn | "Unsupervised learning in production: k-means, DBSCAN and PCA in Rust" |

---

### `training-observability`

| Field | Value |
|---|---|
| Part | Algorithms |
| Purpose | Show that a model's training is observable with the tools the industry already runs |
| Sections | why observability instead of a bespoke logger · the spans the library emits · the metrics it exports · the containers: Jaeger for traces, Loki for logs, podman and docker · reading a training run · the traces of a batch and of a workflow |
| Examples | `scaling_inside_a_cross_validated_pipeline` (the spans of a cross-validation) |
| Figures | the trace tree of a training run (Mermaid) · the container topology (Mermaid) |
| Metadata | description: observing sciencekit training with OpenTelemetry, Jaeger and Loki; keywords: OpenTelemetry, Jaeger, Loki, model observability, tracing |
| Medium | "Your model is a distributed system: tracing ML training with OpenTelemetry" |
| LinkedIn | "Why I did not write a training logger: OpenTelemetry, Jaeger and Loki for ML" |

---

### `lazy-loading-and-data-sources`

| Field | Value |
|---|---|
| Part | Algorithms |
| Purpose | Explain when lazy loading pays and when it does not, with polars and datafusion |
| Sections | the memory problem · eager against lazy · polars: the eager engine · datafusion: the lazy engine · when lazy is the right answer and when it is overhead · observing the load · out-of-core fits |
| Examples | `streaming_scaler_under_a_shifting_distribution` · `k_means_versus_mini_batch_k_means_on_a_stream` · `out_of_core_classification_of_a_stream` |
| Figures | eager against lazy (Mermaid) · the memory profile (Mermaid) |
| Metadata | description: lazy data loading in sciencekit with polars and datafusion; keywords: lazy loading, polars, datafusion, out-of-core, streaming |
| Medium | "Lazy loading for machine learning: when polars and datafusion actually help" |
| LinkedIn | "Datasets bigger than RAM: the lazy path through a Rust ML library" |

---

### `export-and-save`

| Field | Value |
|---|---|
| Part | Algorithms |
| Purpose | The formats, the trade-offs and the round trip |
| Sections | why export at all · safetensors: the internal format · ONNX: the interchange format · which one and when · exporting a pipeline, not only a model · the netron diagrams · loading back |
| Examples | the export sections of `a-complete-example` |
| Figures | the safetensors layout (Mermaid) · the ONNX graph (Mermaid) · the round trip (Mermaid) |
| Metadata | description: exporting sciencekit models to ONNX and safetensors; keywords: ONNX, safetensors, model export, Netron, interop |
| Medium | "ONNX or safetensors? Exporting models from a Rust ML library" |
| LinkedIn | "Shipping a model: ONNX, safetensors and the pipelines in between" |

---

### `inference-of-saved-models`

| Field | Value |
|---|---|
| Part | Algorithms |
| Purpose | Three production shapes of the same inference |
| Sections | loading a saved model · the job batch · the inference API · the observability of inference |
| Subsections | `#### Job batch` — a batch job whose core is inference: load the model, infer over a dataset in batches, add the result as a column, save the output, with lazy loading · `#### API de inferência` — an Actix-web service that receives an HTTP request and answers with the inference · `#### Observabilidade em inferência` — the OTEL metrics of both, the spans per model, the batches and their sizes |
| Examples | `out_of_core_classification_of_a_stream` (the batch shape) |
| Figures | the three shapes (Mermaid) · the batch job flow (Mermaid) |
| Metadata | description: running inference with sciencekit in a batch job, an API and under observability; keywords: model inference, batch inference, Actix-web, Rust inference API |
| Medium | "Three ways to serve a Rust model: batch job, HTTP API and observability" |
| LinkedIn | "From notebook to production: three inference shapes with one Rust model" |

---

### `choosing-the-estimator`

| Field | Value |
|---|---|
| Part | Algorithms |
| Purpose | The map, adapted from scikit-learn's machine-learning map, with the sciencekit names and the honest caveats |
| Sections | how to read the map · classification · regression · clustering · dimensionality reduction · the caveat: the map is a starting point, the chapters are the argument |
| Examples | links to the algorithm chapters |
| Figures | the estimator map (Mermaid) |
| Metadata | description: which sciencekit estimator to choose for a problem; keywords: choosing an estimator, machine learning map, scikit-learn estimator selection |
| Medium | "Which estimator should I use? A map for sciencekit, adapted from scikit-learn" |
| LinkedIn | "The estimator decision, in one map — and when to ignore the map" |

---

### `interoperability`

| Field | Value |
|---|---|
| Part | Algorithms |
| Purpose | The bindings, what each language gets and how to reach them |
| Sections | the C ABI · the bindings: Lua, Go, Python, Julia, JVM, R · the links · what is stable and what is not |
| Examples | none — the chapter collects the links |
| Figures | the binding topology (Mermaid) |
| Metadata | description: the language bindings of sciencekit; keywords: Rust bindings, Python bindings, Go, Julia, Lua, interop |
| Medium | "Using a Rust ML library from Python, Go, Julia and Lua" |
| LinkedIn | "One model, five languages: the binding strategy of a Rust ML library" |

---

### `examples-index`

| Field | Value |
|---|---|
| Part | Closing |
| Purpose | Every example the book uses, findable from one page |
| Sections | the table of examples: id, what it shows, the chapter that uses it, the link to its gallery page · the gallery itself |
| Examples | all the examples cited by the chapters |
| Figures | none |
| Metadata | description: the sciencekit example index; keywords: sciencekit examples, scikit-learn examples, Rust ML examples |
| Medium | "Every sciencekit example, in one list" |
| LinkedIn | "The example library of a Rust ML project, and how it is organized" |
