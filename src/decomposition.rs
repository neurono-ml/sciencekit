//! The `decomposition` family: PCA and truncated SVD.
//!
//! Every item here resolves to `sciencekit_decomposition`.

pub use sciencekit_decomposition::incremental_pca::{
    SKIncrementalPCA, SKIncrementalPCABuilder, SKIncrementalPCAModel,
};
pub use sciencekit_decomposition::pca::{SKPCA, SKPCABuilder, SKPCAModel, SKPCASolver};
pub use sciencekit_decomposition::truncated_svd::{
    SK_TRUNCATED_SVD_PARITY_TOLERANCE, SKTruncatedSVD, SKTruncatedSVDAlgorithm,
    SKTruncatedSVDBuilder, SKTruncatedSVDModel,
};
pub use sciencekit_decomposition::{incremental_pca, pca, truncated_svd};
