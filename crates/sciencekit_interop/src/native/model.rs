//! The concrete native model variants a conversion can produce
//! (spec `native-model-import`).
//!
//! Conversion is a closed set: only model types whose exported state is
//! sufficient to rebuild them appear here. Each variant owns a real fitted
//! model — never a state wrapper — so the caller can predict and, where the
//! algorithm allows it, continue training.

use sciencekit_cluster::kmeans::SKKMeansModel;
use sciencekit_common::SKFloat;
use sciencekit_ensemble::random_forest::{
    SKRandomForestClassifierModel, SKRandomForestRegressorModel,
};
use sciencekit_linear_model::linear_regression::model::SKLinearRegressionModel;
use sciencekit_linear_model::logistic_regression::SKLogisticRegressionModel;
use sciencekit_linear_model::ridge::SKRidgeModel;
use sciencekit_linear_model::sgd_classifier::SKSGDClassifierModel;
use sciencekit_linear_model::sgd_regressor::SKSGDRegressorModel;
use sciencekit_preprocessing::minmax_scaler::model::SKMinMaxScalerModel;
use sciencekit_preprocessing::robust_scaler::model::SKRobustScalerModel;
use sciencekit_preprocessing::standard_scaler::model::SKStandardScalerModel;
use sciencekit_tree::{SKDecisionTreeClassifierModel, SKDecisionTreeRegressorModel};

/// The prediction surface a native model exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKNativeModelKind {
    /// Emits canonical class indices and class probabilities.
    Classifier,
    /// Emits continuous responses.
    Regressor,
    /// Emits transformed feature matrices.
    Transformer,
    /// Emits cluster assignments.
    Clusterer,
}

impl SKNativeModelKind {
    /// The stable name used by coverage tables and diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Self::Classifier => "classifier",
            Self::Regressor => "regressor",
            Self::Transformer => "transformer",
            Self::Clusterer => "clusterer",
        }
    }
}

/// The classifier models the native importer reconstructs.
#[derive(Debug, Clone)]
pub enum SKNativeClassifier<F: SKFloat> {
    /// Logistic regression.
    LogisticRegression(SKLogisticRegressionModel<F>),
    /// Stochastic-gradient classifier.
    StochasticGradient(SKSGDClassifierModel<F>),
    /// A single decision tree.
    DecisionTree(SKDecisionTreeClassifierModel<F>),
    /// A random forest.
    RandomForest(SKRandomForestClassifierModel<F>),
}

/// The regressor models the native importer reconstructs.
#[derive(Debug, Clone)]
pub enum SKNativeRegressor<F: SKFloat> {
    /// Ordinary least squares.
    LinearRegression(SKLinearRegressionModel<F>),
    /// Ridge regression.
    Ridge(SKRidgeModel<F>),
    /// Stochastic-gradient regressor.
    StochasticGradient(SKSGDRegressorModel<F>),
    /// A single decision tree.
    DecisionTree(SKDecisionTreeRegressorModel<F>),
    /// A random forest.
    RandomForest(SKRandomForestRegressorModel<F>),
}

/// The feature transformers the native importer reconstructs.
#[derive(Debug, Clone)]
pub enum SKNativeTransformer<F: SKFloat> {
    /// Standard scaler.
    StandardScaler(SKStandardScalerModel<F>),
    /// Min/max scaler.
    MinMaxScaler(SKMinMaxScalerModel<F>),
    /// Robust scaler.
    RobustScaler(SKRobustScalerModel<F>),
}

/// The clusterers the native importer reconstructs.
#[derive(Debug, Clone)]
pub enum SKNativeClusterer<F: SKFloat> {
    /// K-means.
    KMeans(SKKMeansModel<F>),
}
