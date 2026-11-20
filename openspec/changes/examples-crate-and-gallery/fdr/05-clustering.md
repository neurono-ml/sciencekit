# 05 — Clustering

Clustering is the first unsupervised family, and the first where the *metric* and the
*initialization* decide more than the algorithm. The examples are built to make that
visible: the same data, different geometry, different answer.

---

### `k_means_clustering_on_a_blobs_dataset`

**Origin** — scikit-learn `examples/cluster/plot_kmeans_blobs.py` · *"Demonstration of k-means assumptions"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` — three isotropic clusters of equal variance and equal size, plus one anisotropic and one differently-sized cluster for the counter-examples |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` — k-means minimizes the within-cluster sum of squares, which is *not* scale-invariant: a feature with a large spread dominates the objective |
| Pipeline | `SKStandardScaler → SKKMeans` |
| Spectrum | `number_of_clusters ∈ {2, 3, 4, 5}`; the initialization arm (`k-means++` vs random) and the seed |
| Effect | with three equal isotropic clusters the true structure is recovered; k-means then fails on the anisotropic cluster (it splits it along its long axis), on the differently-sized cluster (it splits the big one), and on clusters of different densities; the initialization only decides *which* local optimum is reached, not whether the data is suitable |
| Output | the cluster assignment per point, the inertia, the number of iterations, and the silhouette score |
| Availability | available now (`sciencekit_cluster`, `sciencekit_preprocessing`) |

---

### `k_means_plus_plus_initialization`

**Origin** — scikit-learn `examples/cluster/plot_kmeans_plusplus.py` · *"An example of K-Means++ initialization"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` — four clusters with very different densities |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKKMeans(initialization)` |
| Spectrum | `initialization ∈ {k_means++, random}` over several seeds |
| Effect | k-means++ chooses the first centre far from the existing ones, so the initial centres are spread out and the algorithm is much less likely to land in a bad local optimum; the random initialization occasionally puts two centres inside one cluster and leaves another cluster to itself |
| Output | the initial centres per arm, the inertia reached, the number of distinct final partitions over the seeds, and the seed-to-seed stability |
| Availability | available now (`sciencekit_cluster`) |

---

### `k_means_versus_mini_batch_k_means_on_a_stream`

**Origin** — scikit-learn `examples/cluster/plot_mini_batch_kmeans.py` · *"Comparison of the K-Means and MiniBatchKMeans clustering algorithms"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` — 100000 samples, generated and consumed as a stream in blocks |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` with `partial_fit` over the stream |
| Pipeline | stream → `partial_fit(SKStandardScaler)` → `SKMiniBatchKMeans` |
| Spectrum | `batch_size ∈ {3, 100, 1000}`; the number of epochs; the batch arm against the one-shot `SKKMeans` |
| Effect | the one-shot k-means sees every sample before moving a centre; the mini-batch version moves a centre after a few samples, so it tracks a drifting distribution but converges to a noisier optimum; the batch size is the trade-off knob between accuracy and adaptation speed |
| Output | the inertia per epoch, the centres after each epoch against the batch one-shot centres, the wall-clock time of both, and the clustering score |
| Availability | available now (`sciencekit_cluster`, `sciencekit_common` streaming contracts) |

---

### `dbscan_clustering_of_non_convex_shapes`

**Origin** — scikit-learn `examples/cluster/plot_dbscan.py` · *"Demo of DBSCAN clustering algorithm"*

| Stage | Recipe |
|---|---|
| Dataset | `make_moons` with noise, plus a uniform ring of noise points around it |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` — `eps` is a distance, so the scale of the data sets its meaning |
| Pipeline | `SKStandardScaler → SKDBSCAN` |
| Spectrum | `eps ∈ {0.05, 0.1, 0.3, 0.5}`; `minimum_samples ∈ {4, 5, 10}`; the search-strategy arm (`BruteForce` vs `KDTree` vs `BallTree`) and the metric arm |
| Effect | `eps` is the radius within which two points are neighbours and `minimum_samples` is how many neighbours make a point a *core* point; a small `eps` shreds the moons into noise, a large one merges everything into one blob; points that are not in any dense region are labelled as noise instead of being forced into a cluster — the property k-means cannot offer |
| Output | the cluster assignment per point with the noise points marked, the number of clusters found, the core-sample indices, and the score against the true labels (ignoring the noise) |
| Availability | available now (`sciencekit_cluster`) |

---

### `density_reachability_of_optics`

**Origin** — scikit-learn `examples/cluster/plot_optics.py` · *"Demo of OPTICS clustering algorithm"*

| Stage | Recipe |
|---|---|
| Dataset | `make_moons` with varying density plus noise |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKDBSCAN(algorithm = optics)` |
| Spectrum | the `eps` arm (absent for a full reachability plot vs present for the DBSCAN-like clustering) |
| Effect | OPTICS computes the *reachability* of every point and draws the cluster structure as a dendrogram-like plot, so clusters of different densities are visible in one picture; a fixed `eps` cannot see two densities at once, because the radius that captures the sparse cluster also swallows the dense one |
| Output | the reachability plot, the cluster boundaries at several density levels, and the number of clusters at each level |
| Availability | planned (`sciencekit_cluster` — the OPTICS variant on top of the implemented DBSCAN) |

---

### `choosing_the_number_of_clusters_with_a_silhouette_analysis`

**Origin** — scikit-learn `examples/cluster/plot_kmeans_silhouette_analysis.py` · *"Selecting the number of clusters with silhouette analysis on KMeans clustering"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` — three well-separated clusters |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKKMeans` for each candidate `k` |
| Spectrum | `number_of_clusters ∈ {2, 3, 4, 5, 6}` |
| Effect | the silhouette of a point measures how close it is to its own cluster compared with the next cluster; the average silhouette is maximized at the true `k`, and the plot shows the "shoulders" where the choice becomes ambiguous |
| Output | the silhouette plot per `k`, the average silhouette per `k`, and the chosen `k` |
| Availability | available now (`sciencekit_cluster`, `sciencekit_metrics` clustering scorers) |

---

### `clustering_under_a_feature_scale_mismatch`

**Origin** — *no scikit-learn origin — a sciencekit-native extension* · *"Why k-means needs scaling"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` in two dimensions, one of them multiplied by 1000 |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` (or not) |
| Pipeline | `SKKMeans` on the raw design vs on the standardized design |
| Spectrum | the scaling arm; `number_of_clusters ∈ {2, 3}` |
| Effect | without scaling the objective is dominated by the wide feature, so the algorithm ignores the other one entirely and cuts the data along an axis that has nothing to do with the cluster structure; with scaling the clusters are recovered — the clearest demonstration of why scaling sits in the pipeline |
| Output | the cluster assignment and the inertia under both arms, and the distance of the two centres from the true centres |
| Availability | available now (`sciencekit_cluster`, `sciencekit_preprocessing`) |

---

### `hierarchical_clustering_of_toy_datasets`

**Origin** — scikit-learn `examples/cluster/plot_linkage_comparison.py` · *"Comparing different hierarchical linkage methods on toy datasets"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` and `make_moons` side by side |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → agglomerative clustering` |
| Spectrum | the linkage arm (`ward`, `complete`, `average`, `single`); the number of clusters cut from the dendrogram |
| Effect | ward merges the clusters that minimize the within-cluster variance increase and therefore prefers spherical clusters; complete merges by the largest diameter and produces compact, flat clusters; single merges by the smallest distance and produces long, chained clusters that bridge two blobs |
| Output | the dendrogram per linkage, the cut at each level, and the adjusted rand index against the true labels |
| Availability | planned (`sciencekit_cluster` — the agglomerative family) |

---

### `density_based_clustering_of_anisotropic_shapes`

**Origin** — scikit-learn `examples/cluster/plot_cluster_comparison.py` · *"Comparing different clustering algorithms on toy datasets"*

| Stage | Recipe |
|---|---|
| Dataset | four toy shapes: isotropic blobs, anisotropic blobs, interleaved moons and a ring of noise |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKKMeans` vs `SKDBSCAN` over the four shapes |
| Spectrum | the estimator arm; the `eps` and `minimum_samples` arms for DBSCAN |
| Effect | the same pair of estimators that look interchangeable on the isotropic shape diverge completely on the other three: k-means is tied to convex, equally-sized clusters, DBSCAN to density; the comparison is the argument for having both |
| Output | the assignment per shape per estimator, the score against the true labels, and the number of clusters found |
| Availability | available now (`sciencekit_cluster`) |

---

### `spectral_clustering_for_image_segmentation`

**Origin** — scikit-learn `examples/cluster/plot_segmentation_toy.py` · *"Spectral clustering for image segmentation"*

| Stage | Recipe |
|---|---|
| Dataset | a synthetic image built from three concentric rings of pixels |
| Feature selection | the pixel coordinates and their colour value |
| Feature engineering | the pixel grid flattened into a design matrix |
| Preprocessing | `SKStandardScaler` on the coordinates |
| Pipeline | `SKStandardScaler → spectral clustering` |
| Spectrum | the number of clusters; the affinity kernel |
| Effect | the rings are not separable by a straight line, so a centroid-based method fails; the spectral embedding places the rings on a line by using the *connectivity* of the graph rather than the coordinates |
| Output | the segmentation mask, the cluster assignment per pixel and the score against the true regions |
| Availability | planned (`sciencekit_cluster` — the spectral family) |

---

### `bisecting_k_means_against_regular_k_means`

**Origin** — scikit-learn `examples/cluster/plot_bisect_kmeans.py` · *"Bisecting K-Means and Regular K-Means Performance Comparison"*

| Stage | Recipe |
|---|---|
| Dataset | `make_blobs` — 10000 samples in three clusters of unequal size |
| Feature selection | none |
| Feature engineering | none |
| Preprocessing | `SKStandardScaler` |
| Pipeline | `SKStandardScaler → SKKMeans` vs the bisecting variant |
| Spectrum | the number of clusters; the number of bisection iterations |
| Effect | bisecting k-means splits the data in two recursively, which converges faster and needs fewer evaluations of the full objective, but the first split is made without knowing the final number of clusters and can be wrong |
| Output | the wall-clock time, the number of iterations, the inertia and the score of both arms |
| Availability | planned (`sciencekit_cluster`) |

---

### `clustering_of_sparse_documents`

**Origin** — scikit-learn `examples/cluster/plot_coin_ward_segmentation.py` · *"Segmenting the picture of greek coins in regions"*

| Stage | Recipe |
|---|---|
| Dataset | a sparse term-document design — 1000 documents × 500 terms with 1 % density |
| Feature selection | the terms above a document-frequency threshold |
| Feature engineering | term-frequency weighting (the raw counts skewed by document length) |
| Preprocessing | the sparse normalization; **never densified** |
| Pipeline | sparse weighting → `SKKMeans` on the sparse design |
| Spectrum | `number_of_clusters ∈ {2, 4, 8}`; the representation arm (sparse vs dense) |
| Effect | the sparse path and the dense path agree within tolerance, and the sparse path never materializes the 1000 × 500 dense matrix; the weighting changes the clusters because it changes what "close" means |
| Output | the cluster assignment per document, the memory footprint of both paths, and the agreement of the assignments |
| Availability | available now (`sciencekit_cluster`, `sciencekit_math` sparse operations) |
