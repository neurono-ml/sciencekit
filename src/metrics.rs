//! The `metrics` family: scores, scorers and streaming accumulators.
//!
//! Every item here resolves to `sciencekit_metrics`. The crate groups its
//! surface by task, so the umbrella keeps those sub-modules addressable and
//! additionally lifts the most-used scores to the family root.

pub use sciencekit_metrics::accumulators::{SKClassificationAccumulator, SKRegressionAccumulator};
pub use sciencekit_metrics::classification::{
    SKAggregatedScores, SKAveraging, SKClassScores, SKConfusionMatrix, SKLabelSpace,
    sk_accuracy_score, sk_aggregate_class_scores, sk_confusion_matrix, sk_f1_score,
    sk_per_class_precision_recall_f1, sk_precision_score, sk_recall_score,
};
pub use sciencekit_metrics::clustering::{
    SKMutualInformationNormalization, sk_adjusted_rand_score, sk_calinski_harabasz_score,
    sk_completeness_score, sk_davies_bouldin_score, sk_homogeneity_score,
    sk_normalized_mutual_information_score, sk_silhouette_samples, sk_silhouette_score,
    sk_v_measure_score,
};
pub use sciencekit_metrics::ranking::{
    SKAucAccumulator, SKPrecisionRecallCurve, SKRankingAveraging, SKRocCurve,
    sk_average_precision_score, sk_balanced_accuracy_score, sk_binary_log_loss, sk_log_loss,
    sk_matthews_correlation_coefficient, sk_multiclass_roc_auc_score,
    sk_one_versus_rest_roc_auc_scores, sk_precision_recall_curve, sk_roc_auc_score, sk_roc_curve,
};
pub use sciencekit_metrics::regression::{
    sk_mean_absolute_error, sk_mean_squared_error, sk_r2_score, sk_root_mean_squared_error,
};
pub use sciencekit_metrics::scorers::{
    SKAccuracyScorer, SKAucScorer, SKAveragePrecisionScorer, SKBalancedAccuracyScorer, SKF1Scorer,
    SKLogLossScorer, SKMatthewsCorrelationScorer, SKMeanAbsoluteErrorScorer,
    SKMeanSquaredErrorScorer, SKPrecisionScorer, SKR2Scorer, SKRecallScorer,
    SKRootMeanSquaredErrorScorer, SKSilhouetteScorer,
};
pub use sciencekit_metrics::{
    accumulators, classification, clustering, ranking, regression, scorers,
};
