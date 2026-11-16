# Review — deterministic-randomness

Reviewer: independent review pass over the `deterministic-randomness` spec, the
`sciencekit_common` generator/derivation implementation and the stochastic estimators. The
code and the openspec plan are merged; this review closes task 3.4.

## Scenario coverage

| Requirement / scenario | Evidence |
| --- | --- |
| Same seed, same stream | `random::generator_tests::two_generators_with_the_same_seed_produce_identical_streams`, `..._a_clone_continues_the_same_stream`, `..._golden_vectors_pin_the_splitmix64_stream` |
| Uniform values are in range | `generator_tests::next_f64_is_uniform_in_the_unit_interval`, `..._uniform_stays_inside_the_requested_interval`, `..._next_index_stays_inside_the_range` |
| Shuffle is a permutation and reproducible | `..._a_shuffle_is_a_permutation`, `..._the_same_seed_produces_the_same_shuffle_order`, `..._different_seeds_produce_different_shuffle_orders`, `..._empty_and_single_element_shuffles_are_trivial` |
| Normal draws | `..._normal_draws_have_zero_mean_and_unit_variance`, `..._two_normal_draws_consume_one_uniform_pair`, `..._the_same_seed_produces_the_same_normal_stream` |
| Sampling without replacement | `..._sampling_without_replacement_returns_unique_indices_in_range`, `..._is_reproducible_and_matches_a_full_sample` |
| Component seeds differ | `derivation_tests::golden_vectors_pin_the_derivation`, `..._derivation_is_pure_and_reproducible` |
| Derivation is order-independent | `derivation_tests` purity/reproducibility plus the ensemble/SGD thread-invariance tests below |
| Default seed is `0` | `sgd_regressor::builder_tests::*` (`assert_eq!(estimator.random_seed(), 0)`), `sgd_classifier::builder_tests` |
| Explicit seed drives every choice | `sgd_regressor::builder_tests` (`..._random_seed(7)` round-trip), `sgd_regressor::dense_tests`, `sgd_classifier::binary_tests`, streaming/sparse streaming tests |
| No ambient randomness | Every stochastic fit takes `random_seed` from the builder; the generator is the only source (`sciencekit_common::random`); golden vectors pin the stream |
| Thread count does not change the result | `sgd_regressor::dense_tests::the_fit_result_is_thread_count_invariant`, `sgd_classifier/regressor::*::the_same_seed_reproduces_and_the_model_serves_threads`, `random_forest::forest_fit_tests::thread_count_does_not_change_fitted_forest`, `gradient_boosting::...::repeated_fits_and_rayon_thread_counts_produce_identical_models`, `histogram_boosting::determinism_tests::hist_determinism_is_thread_count_invariant`, `bagging::..._regressor_fit_is_invariant_across_rayon_thread_counts_and_non_tree_bases` |
| Stable order | Ensembles store members by index; the thread-invariance tests assert identical fitted models, not completion order |
| Documented reproducibility scope | The generator module documents the same-seed guarantee and the cross-platform tolerance; `golden_vectors_pin_...` pins the current version's stream |

## Findings

1. **Minor — cross-backend tolerance.** The spec promises a tolerance rather than
   bit-for-bit equality across platforms/backends; the tests assert exact equality on one
   platform plus documented tolerances in the acceptance fixtures. Acceptable for the
   scope as written.
2. **Minor — golden vectors are version-scoped.** Upgrading the generator requires
   updating the pinned vectors in the same change, which the spec's scope statement covers.

## Verdict

PASS — the generator, derivation and every stochastic estimator on `main` derive from an
explicit seed, and thread-count invariance is proven where parallel reduction could change
the order.
