//! The `math` family: computational kernels and the math-backend abstraction.
//!
//! Every item here resolves to `sciencekit_math`; the umbrella adds no
//! behaviour. The sub-modules ([`backend`], [`kernels`], [`layout`],
//! [`pairwise`], [`solvers`], [`sparse_ops`], [`decomposition`]) stay
//! addressable so a caller can reach an item by its owning module.

pub use sciencekit_math::backend;
pub use sciencekit_math::decomposition;
pub use sciencekit_math::decomposition::{
    SKRandomizedSvdBuilder, SKRandomizedSvdResult, sk_randomized_svd, sk_seeded_projection,
};
pub use sciencekit_math::kernels;
pub use sciencekit_math::kernels::{
    SK_AXIS_SUM_GRAIN, SK_BINARY_COMBINE_GRAIN, SK_ELEMENTWISE_TRANSFORM_GRAIN,
    SK_KERNEL_PARALLEL_GRAIN, SK_SCALE_IN_PLACE_GRAIN, SKKernel, SKKernelGamma, SKKernelParameters,
    sk_axis_min_max, sk_axis_sum, sk_axis_sum_squares, sk_binary_combine, sk_elementwise_transform,
    sk_feature_population_variance, sk_gram_matrix, sk_gram_vector, sk_kernel_matrix,
    sk_kernel_matrix_sequential, sk_linear_kernel_matrix, sk_polynomial_kernel_matrix,
    sk_precomputed_kernel_matrix, sk_radial_basis_function_kernel_matrix, sk_resolve_kernel_gamma,
    sk_resolve_kernel_parameters, sk_scale_in_place, sk_sigmoid_kernel_matrix,
    sk_validate_kernel_hyperparameters, sk_validate_precomputed_query_kernel,
};
pub use sciencekit_math::layout;
pub use sciencekit_math::layout::{
    sk_force_contiguous, sk_is_c_contiguous, sk_is_f_contiguous, sk_memory_layout,
};
pub use sciencekit_math::pairwise;
pub use sciencekit_math::pairwise::{
    SKPairwiseMetric, sk_chebyshev_distance_matrix, sk_cosine_distance_matrix,
    sk_euclidean_distance_matrix, sk_manhattan_distance_matrix, sk_nan_euclidean_distance_matrix,
    sk_pairwise_distance_matrix, sk_sparse_chebyshev_distance_matrix,
    sk_sparse_cosine_distance_matrix, sk_sparse_euclidean_distance_matrix,
    sk_sparse_manhattan_distance_matrix, sk_sparse_squared_euclidean_distance_matrix,
    sk_squared_euclidean_distance_matrix,
};
pub use sciencekit_math::prelude;

pub use sciencekit_math::solvers;
pub use sciencekit_math::solvers::{
    SKCsrNormalEquationOperator, SKLinearOperator, sk_conjugate_gradient,
};
pub use sciencekit_math::sparse_ops;
pub use sciencekit_math::sparse_ops::{sk_csr_dense_product, sk_sparse_product};
pub use sciencekit_math::{
    SKBackendSelection, SKFaerBackend, SKLUDecomposition, SKLeastSquaresSolution, SKMathBackend,
    SKMatrixMultiplyBackend, SKNormKind, SKQRDecomposition, SKSingularValueDecomposition,
    sk_default_math_backend, sk_math_backend_for_kind,
};
