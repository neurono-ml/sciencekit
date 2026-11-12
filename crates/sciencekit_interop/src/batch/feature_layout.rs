//! The requested representation of a materialized feature block (spec
//! `dataset-interop-core`).

/// The representation a conversion should produce.
///
/// Dense is the default; sparse is requested explicitly because the source
/// declares its selected features sparse. Conversion never densifies a sparse
/// selection and never sparsifies a dense one implicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SKFeatureLayout {
    /// A dense, row-major feature matrix.
    #[default]
    Dense,
    /// A sparse compressed-row feature matrix.
    Sparse,
}

impl SKFeatureLayout {
    /// The stable, lower-case layout name used in traces and errors.
    pub fn name(&self) -> &'static str {
        match self {
            SKFeatureLayout::Dense => "dense",
            SKFeatureLayout::Sparse => "sparse",
        }
    }
}
