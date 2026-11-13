//! Pipeline composition (spec `onnx-export`, requirement "Per-family
//! operators", scenario "Pipeline export").
//!
//! A fitted typed pipeline nests each stage state under `pipeline.stage.<i>` and
//! the final estimator under `pipeline.final`. The neutral nesting carries the
//! tensors but not the inner estimator name, so the exporter recovers the
//! family from the documented, stable tensor names and then wires the stage
//! subgraphs in composition order.

use sciencekit_common::{SKError, SKFloat, SKModelState, sk_extract_state, sk_integer_parameter};

use crate::graph::SKOnnxGraphBuilder;

use super::dispatcher::append_estimator;
use super::helpers::unsupported;
use super::stage::{SKOnnxStageKind, SKOnnxStageOutput};

pub(crate) const SK_PIPELINE_CLASSIFIER_NAME: &str = "SKPipelineClassifier";
pub(crate) const SK_PIPELINE_REGRESSOR_NAME: &str = "SKPipelineRegressor";

/// Translate one typed-pipeline state, when the name matches.
pub(crate) fn append_pipeline<F: SKFloat>(
    estimator_name: &str,
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<Option<SKOnnxStageOutput>, SKError> {
    if estimator_name != SK_PIPELINE_CLASSIFIER_NAME && estimator_name != SK_PIPELINE_REGRESSOR_NAME
    {
        return Ok(None);
    }
    let stage_count = sk_integer_parameter(state.metadata(), "number_of_stages", 0).max(0) as usize;
    let mut current = input.to_string();
    let mut width = state.metadata().feature_count();
    for index in 0..stage_count {
        let stage_namespace = format!("pipeline.stage.{index}");
        let stage_name = detect_nested_estimator(state, &stage_namespace).ok_or_else(|| {
            unsupported(
                estimator_name,
                format!("stage {index} does not expose a tensor signature the exporter recognizes"),
            )
        })?;
        let stage_state = sk_extract_state(state, &stage_namespace, stage_name, width)?;
        let output = append_estimator(
            stage_name,
            &stage_state,
            builder,
            &current,
            &format!("{namespace}{stage_namespace}."),
            width,
        )?;
        if output.kind != SKOnnxStageKind::Transformed {
            return Err(unsupported(
                estimator_name,
                format!("stage {index} is not a transformer, so the pipeline cannot continue"),
            ));
        }
        current = output.tensor;
        width = output.width;
    }
    let final_name = detect_nested_estimator(state, "pipeline.final").ok_or_else(|| {
        unsupported(
            estimator_name,
            "the final estimator does not expose a tensor signature the exporter recognizes",
        )
    })?;
    let final_state = sk_extract_state(state, "pipeline.final", final_name, width)?;
    let output = append_estimator(
        final_name,
        &final_state,
        builder,
        &current,
        &format!("{namespace}pipeline.final."),
        width,
    )?;
    if estimator_name == SK_PIPELINE_CLASSIFIER_NAME && output.kind == SKOnnxStageKind::Regression {
        return Err(unsupported(
            estimator_name,
            "a classifier pipeline must end in a classifier",
        ));
    }
    Ok(Some(output))
}

/// Recover a nested estimator's family from its documented tensor names.
fn detect_nested_estimator<F: SKFloat>(
    state: &SKModelState<F>,
    namespace: &str,
) -> Option<&'static str> {
    let prefix = format!("{namespace}.tensor.");
    let carries = |name: &str| state.tensor(&format!("{prefix}{name}")).is_some();
    if carries("logistic.coefficients") {
        return Some("SKLogisticRegression");
    }
    if carries("linear.coefficients") {
        return Some("SKLinearRegression");
    }
    if carries("standard_scaler.mean") {
        return Some("SKStandardScaler");
    }
    if carries("minmax_scaler.minimum") {
        return Some("SKMinMaxScaler");
    }
    if carries("robust_scaler.center") {
        return Some("SKRobustScaler");
    }
    if carries("one_hot.category_counts") {
        return Some("SKOneHotEncoder");
    }
    if carries("pca.components") {
        return Some("SKPCA");
    }
    if carries("truncated_svd.components") {
        return Some("SKTruncatedSVD");
    }
    if carries("sgd_regressor.coefficients") {
        return Some("SKSGDRegressor");
    }
    None
}
