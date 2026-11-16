# Reuse map — grounding the parity program in the core

This document is normative for the parity program: every algorithm change must declare
which core primitives it consumes, and a primitive may only be extended by a foundations
change. It exists because 146 algorithm changes planned independently would each
re-invent the same solver, sampler, metric or accumulator.

## 1. Primitives that already exist and MUST be reused

| Primitive | Location | What it gives | Consumers in the program |
|---|---|---|---|
| Dense algebra backend | `sciencekit_math::backend` (`gemm`, `svd`, `qr`, `cholesky`, `solve`, `solve_triangular`, `eigh`, `lu`, `slogdet`, `pinv`, `inv`) | every factorization and solve the new families need | GP (Cholesky, solve), graphical lasso, CCA/PLS, MDS, factor analysis, LDA/QDA, whitening, mixtures (log-det) |
| Iterative solvers | `sciencekit_math::solvers` (CG, `SKLinearOperator`, `SKCsrNormalEquationOperator`) | matrix-free solves over borrowed sparse designs | GLM/IRLS, NCA, large GP, KernelRidge with Nystroem, spectral eigensolver |
| Randomized SVD | `sciencekit_math::decomposition` | low-rank approximation, projection, QR | sparse PCA, NMF init, random projection, factor-analysis init, topic-model init |
| Pairwise distances | `sciencekit_math::pairwise` (squared Euclidean, cosine, Manhattan, Chebyshev, NaN-Euclidean, dense/sparse dispatch) | one representation-parity distance layer | neighbors, DBSCAN, KNN imputer, manifold, metrics, mean shift, spectral graphs |
| Kernel algebra | `sciencekit_math::kernels` (linear, polynomial, RBF, sigmoid, precomputed, Gram builder) | base kernels + Gram construction | NuSVM, KernelRidge, KernelPCA, Nystroem/RBFSampler, one-class SVM, GP base kernels |
| Streaming/batching contracts | `sciencekit_common::batching` (`SKLazySource`, `SKSupervisedLazySource`, `SKReopenableSource`, `SKMappableSource`, prefetch driver) | out-of-core input paths | every `streaming`, `mappable` and `chunked` change |
| Execution planning + streaming driver | `sciencekit_common::execution` | intent→plan resolution, batch guard, per-pass policy | every operation |
| Deterministic randomness | `sciencekit_common::random` (`SKRandomGenerator`, sub-seeds) | uniform/normal/shuffle/sample without replacement | GMM/ICA/t-SNE init, bootstrap, RANSAC, Theil-Sen, permutation importance, random projection, randomized search |
| Coordinate-descent substrate | `sciencekit_linear_model::coordinate_descent` (accumulator, screening, sparse accumulation, streaming) | L1/L2 penalized solver with sparse streaming | LassoCV, ElasticNetCV, LogisticRegressionCV, multitask variants, path continuation |
| Logistic solver portfolio | `sciencekit_linear_model::logistic::solver` (LBFGS, Newton, SAGA, LIBLINEAR dual/primal) | binary and multinomial penalized fits | GLM/IRLS, MLP/NCA (LBFGS), calibration |
| SGD losses + class weights | `sciencekit_linear_model::sgd` | loss adapters, learning-rate policies | PassiveAggressive, Perceptron, SGDOneClassSVM |
| Neighbor searchers | `sciencekit_neighbors` (brute force, KD-tree, ball-tree, kNN and radius queries, weights) | exact neighbor and radius search | KernelDensity, LOF, MeanShift, spectral/LLE/t-SNE graphs, label propagation, NCA |
| Tree splitter and growth | `sciencekit_tree` (criteria, splitter incl. `SKSplitter::Random`, growth, class weights) | axis-aligned tree building | ExtraTrees (already supports random thresholds), AdaBoost stumps, RandomTreesEmbedding |
| Boosting substrate | `sciencekit_ensemble::boosting` (loss, stage, subsample, early stopping) | stage-wise additive modeling | AdaBoost, gradient-boosting refactors |
| Histogram binning (private) | `sciencekit_ensemble::histogram_boosting::binning` | numeric/categorical edges, bin assignment, missing/unknown bins | promote to `binning-primitives` → KBins, PDP, calibration, QuantileTransformer |
| Bootstrap sampling (private) | `sciencekit_ensemble::bootstrap` (`sk_sample_indices`, per-tree seeds, sparse gather) | seeded bootstrap/subsample without densification | promote to `resampling-primitives` → RANSAC, Theil-Sen, permutation, datasets |
| Metric accumulators + curves | `sciencekit_metrics::accumulators`, `ranking` (ROC/PR curves) | mergeable streaming metrics | calibration, threshold tuning, DET, likelihood ratios, inspection |
| Label table / label space | `sciencekit_common::label_table`, `sciencekit_metrics::classification::SKLabelSpace` | string and integer label canonicalization | LabelEncoder/Binarizer, multiclass strategy, multilabel metrics |
| Pipeline/boxed stages | `sciencekit_pipeline` (`SKBoxedFittedStage`, `SKBoxedFinalModel`, graph nodes) | runtime composition of fitted stages | stacking, voting, multioutput, calibration, column transformer |
| Platt scaling + CV coupling | `sciencekit_svm::probability` (public) | sigmoid calibration from decision values | CalibratedClassifierCV, threshold tuning |
| Incremental scatter/Gram | `sciencekit_decomposition::incremental_pca` (Chan–Golub–LeVeque, compensated CSR Gram) | mergeable second-order statistics | promote to `streaming-moments` → covariance, LDA/QDA, PLS/CCA |
| Interop sources | `sciencekit_interop` (memmap, Arrow, Polars, DataFusion) | file/columnar inputs | every out-of-core path |

## 2. Missing shared primitives — structural changes, implemented first

| Change id | Home | What it must provide | Without it, these changes duplicate |
|---|---|---|---|
| `streaming-moments` | new `sciencekit_stats::moments` | compensated (Neumaier) mean/variance/covariance/scatter accumulators, mergeable, dense+CSR, weighted | covariance (7), LDA/QDA, CCA/PLS (4), naive Bayes (5), feature-selection scores, GLM Gram, factor analysis, Birch, nearest centroid |
| `distribution-and-em-kernels` | `sciencekit_stats::distributions`, `::em` | Gaussian (uni/multi), categorical, Bernoulli, Dirichlet, log-sum-exp, log-likelihood, EM loop with convergence and init | GMM/BayesianGMM, FactorAnalysis, LDA topics, naive Bayes, iterative imputer |
| `robust-statistics-kernels` | `sciencekit_stats::robust` | median/MAD, quantile summaries, trimmed statistics, MCD concentration steps, Mahalanobis | MinCovDet, EllipticEnvelope, TheilSen, RANSAC, robust preprocessing extensions |
| `pairwise-metric-extension` | `sciencekit_math::pairwise` + kernel-matrix functions | haversine, chi², additive chi², paired distances, Laplacian kernel, kernel matrices as functions | metrics pairwise family, KNN/DBSCAN metric registry, spectral affinities, kernel approximation |
| `sparse-symmetric-eigensolver` | `sciencekit_math::solvers::eigen` | Lanczos/LOBPCG over `SKLinearOperator`, smallest/largest, warm start | spectral clustering/biclustering/coclustering, spectral embedding, Isomap, LLE, KernelPCA at scale |
| `graph-construction-and-algorithms` | new `sciencekit_graph` | adjacency/edge-list types, kNN/radius graph builder over neighbor lists, affinity modes (RBF/kNN/precomputed), Laplacian variants, connected components, MST (Prim), diffusion/PageRank | spectral family (3), HDBSCAN, OPTICS, Isomap, LLE, spectral embedding, t-SNE, label propagation/spreading |
| `composable-kernel-algebra` | `sciencekit_math::kernels` (extend) | Sum/Product/Constant/White/Matern/RationalQuadratic/ExpSineSquared/DotProduct kernels, hyperparameters + gradients, log-marginal-likelihood helpers | GaussianProcess C/R, KernelRidge, KernelPCA, Nystroem |
| `penalized-glm-solver` | `sciencekit_linear_model::glm` | link functions (identity/log/logit/inverse), IRLS with weights and L2, deviance, warm start | Poisson/Gamma/Tweedie, quantile, Huber, GLM refactors |
| `regularization-path-solver` | `sciencekit_linear_model::path` | LARS/Lasso path, OMP path, alpha-grid continuation over the coordinate-descent substrate | lars-family, OMP, LassoCV, ElasticNetCV, LogisticRegressionCV, multitask variants |
| `matrix-factorization-kernels` | `sciencekit_decomposition::factorization` | multiplicative updates, HALS, soft-threshold/L1 projection, dictionary update, sparse coding, whitening/decorrelation | NMF, MiniBatchNMF, SparsePCA, MiniBatchSparsePCA, DictionaryLearning family, SparseCoder, FastICA |
| `binning-primitives` | `sciencekit_stats::binning` (promote private histogram binning) | uniform/quantile/k-means edges, streaming quantile (tdigest), bin assignment, categorical bins | KBinsDiscretizer, QuantileTransformer, partial dependence, calibration bins, histogram-boosting refactor |
| `resampling-primitives` | `sciencekit_common::resampling` (promote private bootstrap + shuffle) | bootstrap indices, subsample without replacement, permutation, stratified folds, per-domain seeds | RANSAC, Theil-Sen, permutation importance, random projection, t-SNE, GMM init, random search, datasets, forest/bagging refactor |
| `boxed-predictor-contract` | `sciencekit_common::predictors` | object-safe fitted classifier/regressor: `predict`, `predict_proba`, `decision_function`, importances, safe sharing | stacking, voting, multioutput, calibration, threshold tuning, permutation importance, partial dependence, self-training, iterative imputer |
| `lbfgs-optimizer` | `sciencekit_math::solvers::lbfgs` (extract from the logistic solver) | generic objective/gradient L-BFGS with line search, convergence and warm start | MLP, NCA, GP hyperparameters, GLM, logistic refactor |
| `label-and-indicator-primitives` | `sciencekit_common::labels` (promote `SKLabelSpace`) | integer label encode/decode, dense/CSR indicator matrices, class weights | LabelEncoder, LabelBinarizer, MultiLabelBinarizer, multiclass strategy, multilabel metrics, text vectorizers |
| `text-tokenization-and-vocabulary` | `sciencekit_text::vocabulary` | unicode tokenizer, n-grams, vocabulary, hashing, CSR document-term builder, tf-idf weighting | Count/TF-IDF/Hashing vectorizers, TfidfTransformer, LDA topics, naive Bayes text paths |

## 3. Family → primitive map

| Family / change | Existing primitives consumed | New primitives required |
|---|---|---|
| `preprocessing` completion | sparse ops, tdigest, data views | `binning-primitives` (KBins, Quantile), `streaming-moments` (Power, TargetEncoder), `label-and-indicator-primitives` (Label*, Ordinal) |
| `feature-selection` | pairwise, metrics accumulators | `streaming-moments`, `boxed-predictor-contract`, `resampling-primitives` |
| `naive-bayes` | dense/CSR scans | `streaming-moments`, `distribution-and-em-kernels`, `label-and-indicator-primitives` |
| `linear-model` completion | coordinate descent, logistic solvers, SGD losses | `regularization-path-solver`, `penalized-glm-solver`, `lbfgs-optimizer`, `resampling-primitives`, `robust-statistics-kernels`, `streaming-moments` |
| `model-selection` completion | splitters, shuffling, metrics | `resampling-primitives`, `boxed-predictor-contract`, `binning-primitives` |
| `kernel` / GP | kernel algebra, Cholesky, randomized SVD | `composable-kernel-algebra`, `pairwise-metric-extension`, `lbfgs-optimizer`, `sparse-symmetric-eigensolver` |
| `neural-network` | SGD framework, randomness | `lbfgs-optimizer`, `resampling-primitives` |
| `semi-supervised` | neighbor searchers, pairwise | `graph-construction-and-algorithms`, `boxed-predictor-contract` |
| `text` | sparse ops, CSR | `text-tokenization-and-vocabulary`, `label-and-indicator-primitives` |
| `compose` / meta | pipeline boxed stages | `boxed-predictor-contract` |
| `ensemble` extras | tree splitter (random thresholds), boosting substrate | `boxed-predictor-contract` (voting/stacking), `resampling-primitives` |
| `calibration` / inspection | Platt scaling, accumulators, metrics curves | `boxed-predictor-contract`, `binning-primitives`, `resampling-primitives` |
| `cluster` | neighbor searchers, pairwise, KMeans, DBSCAN | `graph-construction-and-algorithms`, `sparse-symmetric-eigensolver`, `streaming-moments` (Birch), `pairwise-metric-extension` |
| `decomposition` | randomized SVD, incremental scatter, backend | `matrix-factorization-kernels`, `distribution-and-em-kernels`, `composable-kernel-algebra`, `binning-primitives`, `resampling-primitives` |
| `mixture` | backend (Cholesky/slogdet), randomness | `distribution-and-em-kernels`, `streaming-moments`, `resampling-primitives` |
| `discriminant` | pairwise, backend eigh | `streaming-moments`, `pairwise-metric-extension` |
| `covariance` | backend (solve/eigh/slogdet), graphical-lasso solvers | `streaming-moments`, `robust-statistics-kernels`, `pairwise-metric-extension` |
| `cross-decomposition` | SVD, backend | `streaming-moments`, `pairwise-metric-extension`, `sparse-symmetric-eigensolver` |
| `manifold` | neighbor searchers, pairwise, backend eigh | `graph-construction-and-algorithms`, `sparse-symmetric-eigensolver`, `resampling-primitives`, `lbfgs-optimizer` |
| `neighbors` extras | searchers, pairwise | `pairwise-metric-extension`, `graph-construction-and-algorithms` (transformers), `lbfgs-optimizer` (NCA) |
| `datasets` | randomness | `resampling-primitives` |

## 4. Rules

1. **Declare reuse.** Every algorithm change's proposal names the primitives it consumes
   (existing and new). Review rejects an algorithm change that re-implements one.
2. **Primitives are foundations.** A shared primitive may only change through its own
   foundations change; algorithm changes consume it read-only.
3. **Promote, then generalize.** A helper used inside one crate stays private until a
   second consumer appears; at the third consumer its API is generalized. The four
   existing private helpers (`bootstrap`, `histogram binning`, `incremental scatter`,
   `LBFGS`) are promoted because the parity program already gives them 5+ consumers each.
4. **Refactor in the primitive change.** Each primitive change moves its existing private
   implementation to the shared home and rewires the current consumer, keeping the current
   tests unchanged — no behavior drift while the API moves.
5. **Matrix column.** `parity.yaml` rows gain `reuses: [primitive-id, ...]`, checked by the
   matrix checker, so a planned algorithm cannot claim a primitive that does not exist.
6. **No new crate without a foundations change.** `sciencekit_stats`, `sciencekit_graph`
   and `sciencekit_text` are created by their first primitive change, with the crate
   skeleton, module hygiene rules (`mod.rs` as dispatcher only) and CI wiring.

## 5. Effect on the graph

- The primitives are added to Wave 8 (foundations) in their own lanes (`stats`, `graph`,
  plus extensions to `math`, `contracts`, `linear-model`, `decomposition`, `text`).
- Algorithm changes gain hard dependencies on the primitives they consume; the DAG edge is
  the enforcement, the reuse map is the explanation.
- The structural chain grows from 10 to 26 nodes, all before Wave 9.
- Coverage of an algorithm change is not "the estimator exists" but "the estimator exists
  and its row declares the primitives it reuses".

## 6. Risks

- **Over-abstraction.** Mitigated by rule 3: no primitive is designed for a single
  consumer, and each primitive lists at least three consumers from the matrix.
- **Primitive creep delays delivery.** Mitigated by lane parallelism: `math`, `stats`,
  `graph`, `contracts`, `linear-model`, `decomposition` and `text` primitives run
  concurrently; only the first consumer of each waits.
- **Hidden coupling.** A primitive that imports an estimator crate would invert the
  dependency direction; the review rejects it (primitives depend only on `common` and
  `math`).
- **Moving existing code.** Refactors keep the current tests as the contract; a primitive
  change that alters an existing result is a bug, not a refactor.
