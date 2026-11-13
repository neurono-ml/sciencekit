//! The conversion result: a native model plus the metadata that describes it
//! (spec `native-model-import`).

use sciencekit_common::SKFloat;

use crate::native::descriptor::SKNativeModelDescriptor;
use crate::native::model::{
    SKNativeClassifier, SKNativeClusterer, SKNativeRegressor, SKNativeTransformer,
};

/// A converted native model plus the metadata that describes it.
#[derive(Debug, Clone)]
pub enum SKNativeImport<F: SKFloat> {
    /// A reconstructed classifier.
    Classifier {
        /// The fitted classifier.
        model: SKNativeClassifier<F>,
        /// What the conversion preserved.
        descriptor: SKNativeModelDescriptor,
    },
    /// A reconstructed regressor.
    Regressor {
        /// The fitted regressor.
        model: SKNativeRegressor<F>,
        /// What the conversion preserved.
        descriptor: SKNativeModelDescriptor,
    },
    /// A reconstructed feature transformer.
    Transformer {
        /// The fitted transformer.
        model: SKNativeTransformer<F>,
        /// What the conversion preserved.
        descriptor: SKNativeModelDescriptor,
    },
    /// A reconstructed clusterer.
    Clusterer {
        /// The fitted clusterer.
        model: SKNativeClusterer<F>,
        /// What the conversion preserved.
        descriptor: SKNativeModelDescriptor,
    },
}
