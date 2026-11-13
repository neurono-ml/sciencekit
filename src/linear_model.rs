//! The `linear_model` family: linear, regularized, logistic and SGD models.
//!
//! Every item here resolves to `sciencekit_linear_model`. The crate owns its
//! own `SKClassWeight` policy, distinct from the tree/ensemble and SVM policies
//! of the same name, so the umbrella keeps it inside this family.

pub use sciencekit_linear_model::class_weight::{SKClassWeight, sk_expand_class_weights};
pub use sciencekit_linear_model::coordinate_descent::{SKIntoAlphas, SKSelection};
pub use sciencekit_linear_model::elastic_net::{
    SKElasticNet, SKElasticNetBuilder, SKElasticNetModel,
};
pub use sciencekit_linear_model::lasso::{SKLasso, SKLassoBuilder, SKLassoModel};
pub use sciencekit_linear_model::linear_regression::{
    builder::SKLinearRegressionBuilder, core_implementation::SKLinearRegression,
    model::SKLinearRegressionModel,
};
pub use sciencekit_linear_model::logistic::{SKLogisticPenalty, SKLogisticSolver, SKMultiClass};
pub use sciencekit_linear_model::logistic_regression::{
    SKLogisticRegression, SKLogisticRegressionBuilder, SKLogisticRegressionModel,
};
pub use sciencekit_linear_model::ridge::{
    SKIntoRidgeAlphas, SKRidge, SKRidgeBuilder, SKRidgeModel, SKRidgeSolver,
};
pub use sciencekit_linear_model::sgd::{
    SKConvergencePolicy, SKSGDClassifierLoss, SKSGDLearningRate, SKSGDLoss, SKSGDPenalty,
    SKSGDRegressorLoss,
};
pub use sciencekit_linear_model::sgd_classifier::{
    SKSGDClassifier, SKSGDClassifierBuilder, SKSGDClassifierCoefficients, SKSGDClassifierModel,
};
pub use sciencekit_linear_model::sgd_regressor::{
    SKSGDRegressor, SKSGDRegressorBuilder, SKSGDRegressorCoefficients, SKSGDRegressorModel,
};
pub use sciencekit_linear_model::{
    class_weight, coordinate_descent, elastic_net, lasso, linear_regression, logistic,
    logistic_regression, ridge, sgd, sgd_classifier, sgd_regressor,
};
