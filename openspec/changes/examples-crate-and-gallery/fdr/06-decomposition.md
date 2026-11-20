# 06 — Decomposition

Decomposition trades dimensions for structure. The three implemented methods — PCA,
incremental PCA and truncated SVD — are the same idea in three costumes: find the
directions along which the data varies most (PCA), find them incrementally as the data
arrives (incremental PCA), find them without forming the covariance matrix (truncated
SVD).

---

### `principal_component_analysis_on_the_iris_dataset`

**Origin** — scikit-learn `examples/decomposition/plot_pca_iris.py` · *"Principal Component Analysis (PCA) on Iris Dataset"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded) — 150 × 4 |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` — PCA maximizes the variance of the projection, so without scaling the widest feature defines the first component |
| Pipeline | `SKStandardScaler → SKPCA` |
| Spectrum | `number_of_components ∈ {1, 2, 3, 4}` |
| Effect | the first component captures the sepal/petal length contrast — the direction along which the classes separate — and the second captures the petal width; with two components the three classes are already visually separable, and the 2D projection of the 4D data loses almost no information |
| Output | the explained variance and explained variance ratio per component, the projection coordinates, and the 2D scatter with the class labels |
| Availability | available now (`sciencekit_decomposition`, `sciencekit_preprocessing`) |

---

### `choosing_the_number_of_components_by_the_variance_spectrum`

**Origin** — scikit-learn `examples/decomposition/plot_pca_vs_fa_model_selection.py` · *"Model selection with Probabilistic PCA and Factor Analysis (FA)"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 1000 × 50 with a low-rank structure (rank 5) plus noise |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKPCA` |
| Spectrum | `number_of_components ∈ {1..20}` |
| Effect | the explained variance ratio jumps for the first five components and then flattens into a noise floor — the elbow that reveals the true rank; choosing more components than the elbow adds directions that fit noise |
| Output | the scree plot, the cumulative explained variance, and the elbow point |
| Availability | available now (`sciencekit_decomposition`) |

---

### `incremental_pca_on_a_stream`

**Origin** — scikit-learn `examples/decomposition/plot_incremental_pca.py` · *"Incremental PCA"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 20000 samples × 50, consumed as a stream in blocks |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` with `partial_fit` over the stream |
| Pipeline | stream → `partial_fit(SKStandardScaler)` → `partial_fit(SKIncrementalPCA)` |
| Spectrum | the batch size; the number of components; the batch arm against the one-shot `SKPCA` |
| Effect | the batch fit converges towards the one-shot components but never matches them exactly, because each batch updates the components from its own samples; the approximation error shrinks with the batch size — the memory-versus-fidelity trade the algorithm exists for |
| Output | the components of both fits, the angle between the corresponding components, the memory footprint, and the wall-clock time |
| Availability | available now (`sciencekit_decomposition`, `sciencekit_common` streaming contracts) |

---

### `truncated_svd_on_a_sparse_document_design`

**Origin** — scikit-learn `examples/decomposition/plot_faces_decomposition.py` · *"Faces dataset decompositions"* (sparse arm) |

| Stage | Recipe |
|---|---|
| Dataset | a sparse term-document design — 2000 documents × 3000 terms with 0.5 % density |
| Feature selection | the terms above a document-frequency threshold |
| Feature engineering | the centering of the design (the SVD of the *centered* matrix, applied through the linear operator) |
| Preprocessing | the sparse normalization; the design is never densified |
| Pipeline | sparse centering → `SKTruncatedSVD` |
| Spectrum | `number_of_components ∈ {10, 50, 100}` |
| Effect | the truncated SVD is the cheapest way to get the leading directions of a sparse design: it never forms the covariance matrix, which would be dense and enormous; the components are the directions of greatest variation in the term space, and the reconstruction error falls with the number of components |
| Output | the singular values, the explained variance, the reconstruction error per component count, and the peak memory against the dense path |
| Availability | available now (`sciencekit_decomposition`) |

---

### `randomized_svd_for_large_low_rank_designs`

**Origin** — *no scikit-learn origin — a sciencekit-native extension* · *"Randomized SVD"*

| Stage | Recipe |
|---|---|
| Dataset | `make_low_rank_matrix` — 10000 × 500 of rank 10 plus noise |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | none |
| Pipeline | `SKRandomizedSvd` |
| Spectrum | the oversampling parameter; the number of power iterations; the exact arm (`SKTruncatedSVD`) as the reference |
| Effect | the randomized algorithm projects the design onto a small random subspace, computes a small SVD there and refines it with power iterations; the approximation converges to the exact singular vectors as the subspace grows, at a fraction of the cost |
| Output | the singular values against the exact ones, the subspace angle, the wall-clock time of both arms, and the approximation error |
| Availability | available now (`sciencekit_decomposition`) |

---

### `projection_of_a_high_dimensional_space_onto_two_components`

**Origin** — scikit-learn `examples/decomposition/plot_pca_iris.py` · *"PCA: 2D projection of a 4D dataset"*

| Stage | Recipe |
|---|---|
| Dataset | `load_iris` (embedded) |
| Feature selection | none |
| Feature engineering | the projection itself |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKPCA(number_of_components = 2)` |
| Spectrum | the number of components; the arm with and without whitening |
| Effect | the projection preserves the directions of greatest variation, so the class structure survives; the distances in the projection are *not* the original distances — the projection discards the components it dropped, and a point far in the discarded direction can land close to another |
| Output | the 2D coordinates with the class labels, the original and projected pairwise distances for a few probe pairs, and the reconstruction error of the dropped components |
| Availability | available now (`sciencekit_decomposition`) |

---

### `kernel_principal_component_analysis_of_a_nonlinear_structure`

**Origin** — scikit-learn `examples/decomposition/plot_kernel_pca.py` · *"Kernel PCA"*

| Stage | Recipe |
|---|---|
| Dataset | `make_moons` — a structure that no linear projection can straighten |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → kernel PCA` |
| Spectrum | `kernel ∈ {linear, polynomial, radial_basis_function, sigmoid}`; `gamma` for the radial kernel |
| Effect | the linear kernel reproduces ordinary PCA; the radial kernel maps the moons onto a space where they are linearly separable, and the projection straightens them — at the cost of a projection that no longer means a direction in the original space |
| Output | the 2D projection per kernel, the explained variance, and the reconstruction error in the original space |
| Availability | planned (`sciencekit_decomposition` — the kernel PCA on top of the implemented kernels) |

---

### `factor_analysis_with_rotation`

**Origin** — scikit-learn `examples/decomposition/plot_varimax_fa.py` · *"Factor Analysis (with rotation) to visualize patterns"*

| Stage | Recipe |
|---|---|
| Dataset | `make_regression` — 1000 × 20 built from three latent factors with correlated loadings |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → factor analysis` |
| Spectrum | the number of factors; the rotation arm (none vs varimax) |
| Effect | PCA finds the directions of maximum variance, which mix several latent causes; factor analysis models the *covariance* structure and the rotation rotates the factors into a more interpretable basis without changing the model |
| Output | the loadings before and after the rotation, the explained variance per factor, and the reconstructed covariance error |
| Availability | planned (`sciencekit_decomposition` — the factor-analysis family) |

---

### `blind_source_separation_with_fast_ica`

**Origin** — scikit-learn `examples/decomposition/plot_ica_blind_source_separation.py` · *"Blind source separation using FastICA"*

| Stage | Recipe |
|---|---|
| Dataset | three independent source signals mixed by a random invertible matrix, then whitened |
| Feature selection | none |
| Feature engineering | the whitening step (decorrelation) precedes the separation |
| Preprocessing | `SKStandardScaler` plus the whitening |
| Pipeline | whitening → `FastICA` |
| Spectrum | `number_of_components ∈ {2, 3, 5}`; the convergence tolerance |
| Effect | ICA looks for the directions along which the components are *independent* rather than merely decorrelated; it recovers the sources up to a scale and a sign, which is why the recovered signals are compared up to that ambiguity |
| Output | the recovered sources against the originals, the correlation matrix between them, and the iterations to convergence |
| Availability | planned (`sciencekit_decomposition` — the ICA family) |

---

### `sparse_coding_with_a_precomputed_dictionary`

**Origin** — scikit-learn `examples/decomposition/plot_sparse_coding.py` · *"Sparse coding with a precomputed dictionary"*

| Stage | Recipe |
|---|---|
| Dataset | a synthetic image-like signal, 64 × 64 pixels |
| Feature selection | the pixels |
| Feature engineering | the patch extraction that turns the image into a set of patches |
| Preprocessing | `SKStandardScaler` on the patches |
| Pipeline | patch extraction → sparse coding against a fixed dictionary |
| Spectrum | the number of atoms used; the sparsity level |
| Effect | sparse coding reconstructs a signal as a combination of a few dictionary atoms; the more atoms are allowed, the better the reconstruction, and the less sparse the code becomes — the two knobs pull against each other |
| Output | the reconstruction error per sparsity level, the atom usage histogram, and the reconstructed image |
| Availability | planned (`sciencekit_decomposition` — the sparse-coding and dictionary-learning family) |

---

### `non_negative_matrix_factorization_of_a_document_term_matrix`

**Origin** — scikit-learn `examples/applications/plot_topics_extraction_with_nmf_lda.py` · *"Topic extraction with Non-negative Matrix Factorization"* |

| Stage | Recipe |
|---|---|
| Dataset | a sparse term-document design — 3000 documents × 2000 terms |
| Feature selection | the terms above a document-frequency threshold |
| Feature engineering | the term-frequency weighting, applied to a non-negative design |
| Preprocessing | the sparse normalization; the design is never densified and stays non-negative |
| Pipeline | sparse weighting → `NMF` |
| Spectrum | `number_of_components ∈ {5, 10, 20}`; the initialization arm and the seed |
| Effect | NMF constrains the factors to be non-negative, so every component is a *sum* of parts and the result is directly readable as a topic; the initialization decides the local optimum, and the number of components decides the granularity of the topics |
| Output | the component-word tables, the reconstruction error, and the top terms per topic |
| Availability | planned (`sciencekit_decomposition` — the NMF family) |

---

### `manifold_learning_of_a_swiss_roll`

**Origin** — scikit-learn `examples/manifold/plot_swissroll.py` · *"Swiss Roll And Swiss-Hole Reduction"*

| Stage | Recipe |
|---|---|
| Dataset | `make_swiss_roll` — a three-dimensional surface rolled into a spiral, with a hole variant |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → manifold embedding` |
| Spectrum | the method arm (isometric mapping, locally linear embedding, t-distributed stochastic neighbor embedding); the neighbour count; the perplexity for the stochastic embedding |
| Effect | all three preserve the *neighbourhood* structure and not the distances, so the unrolled sheet is geometrically distorted but topologically correct; the hole variant breaks the local connectivity that the neighbourhood graph assumes, which is why the embedding tears there |
| Output | the 2D embedding per method, the trustworthiness against the ground-truth sheet coordinates, and the neighbourhood preservation score |
| Availability | absent (no sciencekit path yet — manifold learning is not on the roadmap) |

---

### `kernel_approximation_of_an_explicit_feature_map`

**Origin** — scikit-learn `examples/miscellaneous/plot_kernel_approximation.py` · *"Explicit feature map approximation for RBF kernels"*

| Stage | Recipe |
|---|---|
| Dataset | `make_classification` — 2000 × 20 |
| Feature selection | none |
| Feature engineering | the random Fourier features *are* the engineered features |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → random Fourier features → SKLogisticRegression` |
| Spectrum | the number of components ∈ {100, 500, 1000}; the kernel arm |
| Effect | the explicit map approximates the kernel's inner product with a finite-dimensional inner product, so any linear model on the mapped data solves the kernel problem approximately; the error falls with the number of components, and the model becomes a plain linear model that trains in linear time |
| Output | the approximation error against the exact kernel, the training time against the kernel machine, and the test accuracy of both arms |
| Availability | planned (`sciencekit_decomposition` — the kernel-approximation family) |
