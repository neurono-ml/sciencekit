//! Preprocessing exporters (spec `onnx-export`, requirement "Per-family
//! operators").
//!
//! A scaler's state is its learned per-column statistics; the graph reproduces
//! scikit-learn's transform with broadcasting `Sub`/`Mul`/`Div` and, for the
//! min-max scaler, `Clip`.

use sciencekit_common::{SKError, SKFloat, SKModelState, SKParameterValue};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode, SKOnnxTensorDType};

use super::helpers::{
    boolean_parameter, floating_tensor, integer_tensor, string_parameter, unsupported,
};
use super::stage::{SKOnnxStageKind, SKOnnxStageOutput};

/// The stable estimator name of the standard scaler.
pub(crate) const SK_STANDARD_SCALER_NAME: &str = "SKStandardScaler";
/// The stable estimator name of the min-max scaler.
pub(crate) const SK_MINMAX_SCALER_NAME: &str = "SKMinMaxScaler";
/// The stable estimator name of the robust scaler.
pub(crate) const SK_ROBUST_SCALER_NAME: &str = "SKRobustScaler";
/// The stable estimator name of the one-hot encoder.
pub(crate) const SK_ONE_HOT_ENCODER_NAME: &str = "SKOneHotEncoder";

/// Translate one preprocessing estimator state, when the name matches.
pub(crate) fn append_preprocessing<F: SKFloat>(
    estimator_name: &str,
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<Option<SKOnnxStageOutput>, SKError> {
    match estimator_name {
        SK_STANDARD_SCALER_NAME => {
            append_standard_scaler(state, builder, input, namespace).map(Some)
        }
        SK_MINMAX_SCALER_NAME => append_minmax_scaler(state, builder, input, namespace).map(Some),
        SK_ROBUST_SCALER_NAME => append_robust_scaler(state, builder, input, namespace).map(Some),
        SK_ONE_HOT_ENCODER_NAME => {
            append_one_hot_encoder(state, builder, input, namespace).map(Some)
        }
        _ => Ok(None),
    }
}

fn append_standard_scaler<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let (dimensions, mean) = floating_tensor(state, "standard_scaler.mean")?;
    let (_, scale) = floating_tensor(state, "standard_scaler.scale")?;
    let width = dimensions.first().copied().unwrap_or(0);
    let mut current = input.to_string();
    if boolean_parameter(state, "with_mean", true) {
        builder.add_floating_initializer(
            format!("{namespace}standard_scaler.mean"),
            vec![width],
            &mean,
        );
        let centered = format!("{namespace}standard_scaler.centered");
        builder.add_node(SKOnnxNode::new(
            "Sub",
            vec![current, format!("{namespace}standard_scaler.mean")],
            vec![centered.clone()],
        ));
        current = centered;
    }
    if boolean_parameter(state, "with_std", true) {
        builder.add_floating_initializer(
            format!("{namespace}standard_scaler.scale"),
            vec![width],
            &scale,
        );
        let scaled = format!("{namespace}standard_scaler.scaled");
        builder.add_node(SKOnnxNode::new(
            "Div",
            vec![current, format!("{namespace}standard_scaler.scale")],
            vec![scaled.clone()],
        ));
        current = scaled;
    }
    Ok(SKOnnxStageOutput {
        tensor: current,
        width,
        kind: SKOnnxStageKind::Transformed,
    })
}

fn append_minmax_scaler<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let (dimensions, minimum) = floating_tensor(state, "minmax_scaler.minimum")?;
    let (_, maximum) = floating_tensor(state, "minmax_scaler.maximum")?;
    let (_, scale) = floating_tensor(state, "minmax_scaler.scale")?;
    let width = dimensions.first().copied().unwrap_or(0);
    let feature_minimum = super::helpers::float_parameter(state, "feature_minimum", 0.0);
    let feature_maximum = super::helpers::float_parameter(state, "feature_maximum", 1.0);

    builder.add_floating_initializer(
        format!("{namespace}minmax_scaler.minimum"),
        vec![width],
        &minimum,
    );
    builder.add_floating_initializer(
        format!("{namespace}minmax_scaler.scale"),
        vec![width],
        &scale,
    );
    builder.add_floating_initializer(
        format!("{namespace}minmax_scaler.feature_minimum"),
        Vec::new(),
        &[F::from(feature_minimum).unwrap_or_else(F::zero)],
    );
    let centered = format!("{namespace}minmax_scaler.centered");
    builder.add_node(SKOnnxNode::new(
        "Sub",
        vec![
            input.to_string(),
            format!("{namespace}minmax_scaler.minimum"),
        ],
        vec![centered.clone()],
    ));
    let scaled = format!("{namespace}minmax_scaler.scaled");
    builder.add_node(SKOnnxNode::new(
        "Mul",
        vec![centered, format!("{namespace}minmax_scaler.scale")],
        vec![scaled.clone()],
    ));
    let shifted = format!("{namespace}minmax_scaler.shifted");
    builder.add_node(SKOnnxNode::new(
        "Add",
        vec![scaled, format!("{namespace}minmax_scaler.feature_minimum")],
        vec![shifted.clone()],
    ));
    let mut current = shifted;
    if boolean_parameter(state, "clip", false) {
        builder.add_floating_initializer(
            format!("{namespace}minmax_scaler.clip_minimum"),
            Vec::new(),
            &[F::from(feature_minimum).unwrap_or_else(F::zero)],
        );
        builder.add_floating_initializer(
            format!("{namespace}minmax_scaler.clip_maximum"),
            Vec::new(),
            &[F::from(feature_maximum).unwrap_or_else(F::zero)],
        );
        let clipped = format!("{namespace}minmax_scaler.clipped");
        builder.add_node(SKOnnxNode::new(
            "Clip",
            vec![
                current,
                format!("{namespace}minmax_scaler.clip_minimum"),
                format!("{namespace}minmax_scaler.clip_maximum"),
            ],
            vec![clipped.clone()],
        ));
        current = clipped;
    }
    let _ = maximum;
    Ok(SKOnnxStageOutput {
        tensor: current,
        width,
        kind: SKOnnxStageKind::Transformed,
    })
}

fn append_robust_scaler<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let (dimensions, center) = floating_tensor(state, "robust_scaler.center")?;
    let (_, scale) = floating_tensor(state, "robust_scaler.scale")?;
    let width = dimensions.first().copied().unwrap_or(0);
    let mut current = input.to_string();
    if boolean_parameter(state, "with_centering", true) {
        builder.add_floating_initializer(
            format!("{namespace}robust_scaler.center"),
            vec![width],
            &center,
        );
        let centered = format!("{namespace}robust_scaler.centered");
        builder.add_node(SKOnnxNode::new(
            "Sub",
            vec![current, format!("{namespace}robust_scaler.center")],
            vec![centered.clone()],
        ));
        current = centered;
    }
    if boolean_parameter(state, "with_scaling", true) {
        builder.add_floating_initializer(
            format!("{namespace}robust_scaler.scale"),
            vec![width],
            &scale,
        );
        let scaled = format!("{namespace}robust_scaler.scaled");
        builder.add_node(SKOnnxNode::new(
            "Div",
            vec![current, format!("{namespace}robust_scaler.scale")],
            vec![scaled.clone()],
        ));
        current = scaled;
    }
    Ok(SKOnnxStageOutput {
        tensor: current,
        width,
        kind: SKOnnxStageKind::Transformed,
    })
}

fn append_one_hot_encoder<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    if string_parameter(state, "handle_unknown", "reject") == "reject" {
        return Err(unsupported(
            SK_ONE_HOT_ENCODER_NAME,
            "the reject policy raises at run time, which an exported graph cannot express; export with handle_unknown(ignore)",
        ));
    }
    let counts = integer_tensor(state, "one_hot.category_counts")?;
    let kinds = state
        .metadata()
        .parameter("one_hot.column_kinds")
        .and_then(SKParameterValue::as_list)
        .ok_or(SKError::InvalidHyperparameter {
            name: "one_hot.column_kinds",
            reason: "the exported one-hot state carries no column kinds".to_string(),
        })?;
    let mut categories_by_column = Vec::with_capacity(counts.len());
    for (column, kind) in kinds.iter().enumerate() {
        let kind = kind.as_string().unwrap_or("nominal");
        if kind != "integer" {
            return Err(unsupported(
                SK_ONE_HOT_ENCODER_NAME,
                format!(
                    "column {column} is nominal (string categories); the ONNX exporter currently maps numeric integer categories only"
                ),
            ));
        }
        let codes = state
            .metadata()
            .parameter(&format!("one_hot.integer_categories.{column}"))
            .and_then(SKParameterValue::as_list)
            .ok_or(SKError::InvalidHyperparameter {
                name: "one_hot.integer_categories",
                reason: format!("the state carries no codes for column {column}"),
            })?;
        // Keep the declared category count authoritative: the code table may be
        // sparse for integer tables.
        let declared = counts[column].max(0) as usize;
        let mut values = vec![0.0_f64; declared];
        for (index, code) in codes.iter().enumerate().take(declared) {
            values[index] = code.as_integer().unwrap_or(0) as f64;
        }
        categories_by_column.push(values);
    }
    let mut pieces = Vec::new();
    let mut width = 0;
    for (column, codes) in categories_by_column.iter().enumerate() {
        let gathered = format!("{namespace}one_hot.column.{column}");
        builder.add_node(
            SKOnnxNode::new(
                "Gather",
                vec![
                    input.to_string(),
                    format!("{namespace}one_hot.index.{column}"),
                ],
                vec![gathered.clone()],
            )
            .attribute_integer("axis", 1),
        );
        builder.add_integer_initializer(
            format!("{namespace}one_hot.index.{column}"),
            vec![1],
            &[column as i64],
        );
        let mut columns = Vec::new();
        for (category, code) in codes.iter().enumerate() {
            builder.add_floating_initializer(
                format!("{namespace}one_hot.code.{column}.{category}"),
                Vec::new(),
                &[F::from(*code).unwrap_or_else(F::zero)],
            );
            let compared = format!("{namespace}one_hot.equal.{column}.{category}");
            builder.add_node(SKOnnxNode::new(
                "Equal",
                vec![
                    gathered.clone(),
                    format!("{namespace}one_hot.code.{column}.{category}"),
                ],
                vec![compared.clone()],
            ));
            let indicator = format!("{namespace}one_hot.indicator.{column}.{category}");
            builder.add_node(
                SKOnnxNode::new("Cast", vec![compared], vec![indicator.clone()])
                    .attribute_integer("to", indicator_element_type::<F>()),
            );
            columns.push(indicator);
        }
        width += columns.len();
        pieces.extend(columns);
    }
    let encoded = format!("{namespace}one_hot.encoded");
    builder.add_node(
        SKOnnxNode::new("Concat", pieces, vec![encoded.clone()]).attribute_integer("axis", 1),
    );
    Ok(SKOnnxStageOutput {
        tensor: encoded,
        width,
        kind: SKOnnxStageKind::Transformed,
    })
}

fn indicator_element_type<F: SKFloat>() -> i64 {
    SKOnnxTensorDType::for_scalar::<F>().element_type() as i64
}
