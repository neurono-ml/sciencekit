//! The traits, types and estimators that typical code needs.
//!
//! The prelude is deliberately small: the shared vocabulary plus one or two
//! headline estimators per family. It is a curated list, not a glob import —
//! `use sciencekit::prelude::*` cannot silently pull every model in, and it
//! cannot fail on a name that two families both export.
//!
//! # What is deliberately absent
//!
//! - `SKClassWeight`: three distinct policies share the name (tree/ensemble,
//!   linear-model logistic and SVM). Import the one you mean from its family
//!   module.
//! - `SKSplitter`: a split-selection enum in [`crate::tree`] and the train/test
//!   partition trait in [`crate::model_selection`]. Neither is a prelude name.
//! - The full model catalogue: reach for [`crate::ensemble`],
//!   [`crate::decomposition`] or any other family module when you need more.
//! - Engine adapters and device contracts: those live in [`crate::interop`] and
//!   [`crate::accelerators`].

// Shared vocabulary.
pub use crate::{
    SKBackendKind, SKBuilder, SKBuilderState, SKClassifierPredictor, SKClusterIdentifier,
    SKClusterLabel, SKClustererPredictor, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExportable, SKFeatureBlock, SKFeatureTransformer, SKFloat, SKImportable, SKModelState,
    SKOperationKind, SKOutlierLabel, SKOutlierPredictor, SKRegressorPredictor, SKSupervisedFit,
    SKTargetView, SKUnsupervisedFit,
};

// Preprocessing and imputation.
pub use crate::impute::SKSimpleImputer;
pub use crate::preprocessing::{
    SKMinMaxScaler, SKOneHotEncoder, SKPolynomialFeatures, SKRobustScaler, SKStandardScaler,
    SKStandardScalerModel,
};

// Supervised learning.
pub use crate::ensemble::{
    SKGradientBoostingClassifier, SKGradientBoostingRegressor, SKRandomForestClassifier,
    SKRandomForestRegressor,
};
pub use crate::linear_model::{
    SKElasticNet, SKLasso, SKLinearRegression, SKLinearRegressionModel, SKLogisticRegression,
    SKRidge, SKSGDClassifier, SKSGDRegressor,
};
pub use crate::neighbors::{SKKNeighborsClassifier, SKKNeighborsRegressor};
pub use crate::svm::{SKLinearSVC, SKSVC, SKSVR};
pub use crate::tree::{
    SKDecisionTreeClassifier, SKDecisionTreeRegressor, SKMaximumFeatures, SKTreeCriterion,
};

// Unsupervised learning.
pub use crate::cluster::{SKDBSCAN, SKKMeans, SKMiniBatchKMeans};
pub use crate::decomposition::{SKIncrementalPCA, SKPCA, SKTruncatedSVD};
pub use crate::outlier::SKIsolationForest;

// Composition and evaluation.
pub use crate::metrics::{
    SKConfusionMatrix, sk_accuracy_score, sk_f1_score, sk_mean_squared_error, sk_r2_score,
};
pub use crate::model_selection::{
    SKGroupKFold, SKKFold, SKStratifiedKFold, SKTrainTestSplit, sk_train_test_split,
};
pub use crate::pipeline::{SKPipeline, SKPipelineClassifier, SKPipelineRegressor};
