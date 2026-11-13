//! Compile-time proof that the umbrella re-exports the stable surface of every
//! sub-crate under documented module paths.
//!
//! A missing module, a renamed item or an unresolved collision fails the build,
//! so this companion test is the executable form of the "Common types resolve"
//! and "Sub-crate modules remain addressable" scenarios.

fn sized<T>() {}
fn implements<T: crate::SKFloat>() {}

#[test]
fn every_family_module_is_addressable() {
    #[cfg(feature = "telemetry")]
    #[allow(unused_imports)]
    use crate::telemetry;
    #[allow(unused_imports)]
    use crate::{
        accelerators, cluster, decomposition, ensemble, ffi, impute, interop, linear_model, math,
        metrics, model_selection, neighbors, outlier, pipeline, preprocessing, svm, tree,
    };
}

#[test]
fn common_types_resolve_at_the_root() {
    sized::<crate::SKError>();
    sized::<crate::SKBuilderState>();
    sized::<crate::SKExecutionContext>();
    sized::<crate::SKExecutionMode>();
    sized::<crate::SKBackendKind>();
    sized::<crate::SKModelState<f64>>();
    sized::<crate::SKTensor<f64>>();
    sized::<crate::SKOperationAttributes>();
    sized::<crate::SKDataView<'static, f64>>();
    sized::<crate::SKTargetView<'static, f64>>();
    implements::<f64>();
}

#[test]
fn common_traits_resolve_at_the_root() {
    fn supervised<T: crate::SKSupervisedFit<f64>>() {}
    fn unsupervised<T: crate::SKUnsupervisedFit<f64>>() {}
    fn exporter<T: crate::SKExportable<f64>>() {}
    let _ = supervised::<crate::linear_model::SKLinearRegression>;
    let _ = unsupervised::<crate::preprocessing::SKStandardScaler>;
    let _ = exporter::<crate::preprocessing::SKStandardScalerModel<f64>>;
}

#[test]
fn preprocessing_and_imputation_types_resolve() {
    sized::<crate::preprocessing::SKStandardScaler>();
    sized::<crate::preprocessing::SKStandardScalerBuilder>();
    sized::<crate::preprocessing::SKStandardScalerModel<f64>>();
    sized::<crate::preprocessing::SKMinMaxScaler<f64>>();
    sized::<crate::preprocessing::SKRobustScaler<f64>>();
    sized::<crate::preprocessing::SKOneHotEncoder>();
    sized::<crate::preprocessing::SKPolynomialFeatures>();
    sized::<crate::impute::SKSimpleImputer<f64>>();
    sized::<crate::impute::SKMissingIndicator<f64>>();
    sized::<crate::impute::SKKNNImputer<f64>>();
}

#[test]
fn supervised_learning_types_resolve() {
    sized::<crate::linear_model::SKLinearRegression>();
    sized::<crate::linear_model::SKRidge<f64>>();
    sized::<crate::linear_model::SKLasso<f64>>();
    sized::<crate::linear_model::SKElasticNet<f64>>();
    sized::<crate::linear_model::SKLogisticRegression<f64>>();
    sized::<crate::linear_model::SKSGDClassifier<f64>>();
    sized::<crate::linear_model::SKSGDRegressor<f64>>();
    sized::<crate::neighbors::SKKNeighborsClassifier<f64>>();
    sized::<crate::neighbors::SKKNeighborsRegressor<f64>>();
    sized::<crate::svm::SKSVC<f64>>();
    sized::<crate::svm::SKSVR<f64>>();
    sized::<crate::svm::SKLinearSVC<f64>>();
    sized::<crate::tree::SKDecisionTreeClassifier<f64>>();
    sized::<crate::tree::SKDecisionTreeRegressor<f64>>();
    sized::<crate::ensemble::SKRandomForestClassifier<f64>>();
    sized::<crate::ensemble::SKRandomForestRegressor<f64>>();
    sized::<crate::ensemble::SKGradientBoostingClassifier<f64>>();
    sized::<crate::ensemble::SKHistGradientBoostingClassifier<f64>>();
}

#[test]
fn unsupervised_and_selection_types_resolve() {
    sized::<crate::cluster::SKKMeans<f64>>();
    sized::<crate::cluster::SKMiniBatchKMeans<f64>>();
    sized::<crate::cluster::SKDBSCAN<f64>>();
    sized::<crate::decomposition::SKPCA<f64>>();
    sized::<crate::decomposition::SKTruncatedSVD<f64>>();
    sized::<crate::decomposition::SKIncrementalPCA<f64>>();
    sized::<crate::outlier::SKIsolationForest<f64>>();
    sized::<crate::model_selection::SKTrainTestSplit<f64>>();
    sized::<crate::model_selection::SKKFold>();
    sized::<crate::pipeline::SKPipeline>();
    sized::<crate::pipeline::SKPipelineGraph<f64>>();
    sized::<crate::model_selection::SKPartitionSize>();
}

#[test]
fn metrics_math_and_integration_types_resolve() {
    sized::<crate::metrics::classification::SKConfusionMatrix>();
    let _ = crate::metrics::regression::sk_mean_squared_error::<f64>;
    sized::<crate::math::SKFaerBackend>();
    let _ = crate::math::sk_pairwise_distance_matrix::<f64>;
    sized::<crate::accelerators::SKDeviceBuffer<f64>>();
    sized::<crate::accelerators::SKOpenClBackend>();
    #[cfg(feature = "telemetry")]
    sized::<crate::telemetry::SKTelemetryConfiguration>();
    #[cfg(feature = "onnx")]
    sized::<crate::interop::SKOnnxModel>();
    sized::<crate::ffi::SKFfiStatus>();
}

#[test]
fn colliding_names_live_in_their_family_modules() {
    // Each family keeps its own `SKClassWeight`: the tree/ensemble policy, the
    // linear-model logistic policy and the SVM policy are distinct types.
    sized::<crate::tree::SKClassWeight<f64>>();
    sized::<crate::linear_model::SKClassWeight<f64>>();
    sized::<crate::svm::SKClassWeight<f64>>();

    // `SKSplitter` is a second, lower-risk collision: a split-selection enum in
    // `tree` and a train/test partition trait in `model_selection`. The prelude
    // exports neither, so the bare name never reaches a program unqualified.
    sized::<crate::tree::SKSplitter>();
    fn partition_strategy<T: crate::model_selection::SKSplitter<f64>>() {}
    let _ = partition_strategy::<crate::model_selection::SKKFold>;

    // `SKFeatureBlock` and the device vocabulary are single types re-exported by
    // several modules; the same identity is what makes the duplication safe.
    sized::<crate::SKFeatureBlock<f64>>();
    sized::<crate::pipeline::SKFeatureBlock<f64>>();
    sized::<crate::SKDeviceInfo>();
    sized::<crate::accelerators::SKDeviceInfo>();
}

/// Compile-time proof that a name re-exported through two paths is one type.
///
/// Each function body only type-checks when the two spellings denote the same
/// type; a second definition in either crate would fail the build.
#[allow(dead_code)]
mod shared_identities_are_one_type {
    pub(crate) fn feature_block(
        value: crate::SKFeatureBlock<f64>,
    ) -> crate::pipeline::SKFeatureBlock<f64> {
        value
    }

    fn into_feature_block<T>()
    where
        T: crate::SKIntoFeatureBlock<f64> + crate::pipeline::SKIntoFeatureBlock<f64>,
    {
    }

    fn device_info(value: crate::SKDeviceInfo) -> crate::accelerators::SKDeviceInfo {
        value
    }

    fn device_capabilities(
        value: crate::SKDeviceCapabilities,
    ) -> crate::accelerators::SKDeviceCapabilities {
        value
    }

    fn gpu_kernel(value: crate::SKGpuKernel) -> crate::accelerators::SKGpuKernel {
        value
    }

    fn fallback_reason(value: crate::SKFallbackReason) -> crate::accelerators::SKFallbackReason {
        value
    }

    fn outlier_label(value: crate::SKOutlierLabel) -> crate::outlier::SKOutlierLabel {
        value
    }

    fn outlier_predictor<
        T: crate::SKOutlierPredictor<f64> + crate::outlier::SKOutlierPredictor<f64>,
    >() {
    }

    fn kernel(value: crate::math::SKKernel) -> crate::svm::SKKernel {
        value
    }

    fn kernel_gamma(value: crate::math::SKKernelGamma<f64>) -> crate::svm::SKKernelGamma<f64> {
        value
    }

    fn kernel_parameters(
        value: crate::math::SKKernelParameters<f64>,
    ) -> crate::svm::SKKernelParameters<f64> {
        value
    }

    fn tree_criterion(value: crate::tree::SKTreeCriterion) -> crate::ensemble::SKTreeCriterion {
        value
    }

    fn maximum_features(
        value: crate::tree::SKMaximumFeatures<f64>,
    ) -> crate::ensemble::SKMaximumFeatures<f64> {
        value
    }

    fn class_weight(value: crate::tree::SKClassWeight<f64>) -> crate::ensemble::SKClassWeight<f64> {
        value
    }
}

#[test]
fn shared_identities_are_one_type() {
    // The proof is compile-time; referencing the module keeps it linked.
    let _ = shared_identities_are_one_type::feature_block;
}

#[test]
fn allocator_selection_vocabulary_resolves() {
    sized::<crate::SKAllocatorKind>();
    sized::<crate::allocator::SKAllocatorKind>();
    let _ = crate::sk_allocator_kind();
}

#[test]
fn the_math_prelude_module_is_addressable() {
    sized::<crate::math::prelude::SKFaerBackend>();
}

#[test]
fn the_low_rank_adapter_api_resolves() {
    sized::<crate::linear_model::SKLoraAdapter<f64>>();
    sized::<crate::linear_model::SKLoraAdapterBuilder<f64>>();
    sized::<crate::linear_model::SKLoraAdapterArtifact<f64>>();
    sized::<crate::linear_model::SKLoraAdapterRegistry<f64>>();
    sized::<crate::linear_model::SKLoraAttachedModel<f64>>();
    sized::<crate::linear_model::SKLoraBaseModel<f64>>();
    sized::<crate::linear_model::SKLoraMergedModel<f64>>();
    sized::<crate::linear_model::SKLoraTargetModule>();
    sized::<crate::linear_model::adapters::lora::SKLoraAdapter<f64>>();
    let _ = crate::linear_model::SK_LORA_ESTIMATOR_NAME;
    let _ = crate::linear_model::SK_LORA_FORMAT_VERSION;
    let _ = crate::linear_model::sk_lora_a_tensor_name;
    let _ = crate::linear_model::sk_lora_b_tensor_name;
}
