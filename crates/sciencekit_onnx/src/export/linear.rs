//! Linear and logistic model exporters (spec `onnx-export`, requirement
//! "Per-family operators").
//!
//! A fitted linear model is a solved coefficient vector; the graph is the
//! decision function `X · Wᵀ + b` written with `MatMul` and `Add`. A logistic
//! model adds the link operator (`Sigmoid` or `Softmax`) and `ArgMax`.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::helpers::{boolean_parameter, floating_tensor, string_parameter, unsupported};
use super::stage::{SKOnnxStageKind, SKOnnxStageOutput};

pub(crate) const SK_LINEAR_REGRESSION_NAME: &str = "SKLinearRegression";
pub(crate) const SK_SGD_REGRESSOR_NAME: &str = "SKSGDRegressor";
pub(crate) const SK_RIDGE_NAME: &str = "SKRidge";
pub(crate) const SK_LASSO_NAME: &str = "SKLasso";
pub(crate) const SK_ELASTIC_NET_NAME: &str = "SKElasticNet";
pub(crate) const SK_LOGISTIC_REGRESSION_NAME: &str = "SKLogisticRegression";

/// Translate one linear or logistic estimator state, when the name matches.
pub(crate) fn append_linear<F: SKFloat>(
    estimator_name: &str,
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<Option<SKOnnxStageOutput>, SKError> {
    let output = match estimator_name {
        SK_LINEAR_REGRESSION_NAME => Some(append_single_vector_regressor(
            state,
            builder,
            input,
            namespace,
            "linear.coefficients",
            "linear.intercept",
        )?),
        SK_SGD_REGRESSOR_NAME => Some(append_single_vector_regressor(
            state,
            builder,
            input,
            namespace,
            "sgd_regressor.coefficients",
            "sgd_regressor.intercept",
        )?),
        SK_RIDGE_NAME => Some(append_multi_alpha_regressor(
            state, builder, input, namespace, "ridge",
        )?),
        SK_LASSO_NAME => Some(append_multi_alpha_regressor(
            state, builder, input, namespace, "lasso",
        )?),
        SK_ELASTIC_NET_NAME => Some(append_multi_alpha_regressor(
            state,
            builder,
            input,
            namespace,
            "elastic_net",
        )?),
        SK_LOGISTIC_REGRESSION_NAME => Some(append_logistic_regression(
            state, builder, input, namespace,
        )?),
        _ => None,
    };
    Ok(output)
}

/// Append a regressor whose state carries one coefficient vector.
fn append_single_vector_regressor<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
    coefficient_tensor: &str,
    intercept_tensor: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let (_, coefficients) = floating_tensor(state, coefficient_tensor)?;
    let (_, intercept) = floating_tensor(state, intercept_tensor)?;
    let [intercept] = intercept.as_slice() else {
        return Err(SKError::ShapeMismatch {
            expected: vec![1],
            found: vec![intercept.len()],
        });
    };
    let width = coefficients.len();
    builder.add_floating_initializer(
        format!("{namespace}linear.coefficients"),
        vec![width, 1],
        &coefficients,
    );
    let product = format!("{namespace}linear.product");
    builder.add_node(SKOnnxNode::new(
        "MatMul",
        vec![input.to_string(), format!("{namespace}linear.coefficients")],
        vec![product.clone()],
    ));
    let mut current = product;
    if boolean_parameter(state, "include_intercept", true) {
        builder.add_floating_initializer(
            format!("{namespace}linear.intercept"),
            vec![1, 1],
            &[*intercept],
        );
        let shifted = format!("{namespace}linear.prediction");
        builder.add_node(SKOnnxNode::new(
            "Add",
            vec![current, format!("{namespace}linear.intercept")],
            vec![shifted.clone()],
        ));
        current = shifted;
    }
    Ok(SKOnnxStageOutput {
        tensor: current,
        width: 1,
        kind: SKOnnxStageKind::Regression,
    })
}

/// Append a coordinate-descent family regressor, requiring a single alpha.
fn append_multi_alpha_regressor<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
    prefix: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let (_, alphas) = floating_tensor(state, &format!("{prefix}.alphas"))?;
    if alphas.len() != 1 {
        return Err(unsupported(
            prefix,
            "several alphas were fitted; the exported graph is only unambiguous for a single configured alpha",
        ));
    }
    let (shape, coefficients) = floating_tensor(state, &format!("{prefix}.coefficients"))?;
    let (_, intercepts) = floating_tensor(state, &format!("{prefix}.intercepts"))?;
    let [intercept] = intercepts.as_slice() else {
        return Err(SKError::ShapeMismatch {
            expected: vec![1],
            found: vec![intercepts.len()],
        });
    };
    let width = *shape.last().unwrap_or(&coefficients.len());
    if coefficients.len() != width {
        return Err(SKError::ShapeMismatch {
            expected: vec![1, width],
            found: shape,
        });
    }
    builder.add_floating_initializer(
        format!("{namespace}linear.coefficients"),
        vec![width, 1],
        &coefficients,
    );
    let product = format!("{namespace}linear.product");
    builder.add_node(SKOnnxNode::new(
        "MatMul",
        vec![input.to_string(), format!("{namespace}linear.coefficients")],
        vec![product.clone()],
    ));
    let mut current = product;
    if boolean_parameter(state, "include_intercept", true) {
        builder.add_floating_initializer(
            format!("{namespace}linear.intercept"),
            vec![1, 1],
            &[*intercept],
        );
        let shifted = format!("{namespace}linear.prediction");
        builder.add_node(SKOnnxNode::new(
            "Add",
            vec![current, format!("{namespace}linear.intercept")],
            vec![shifted.clone()],
        ));
        current = shifted;
    }
    Ok(SKOnnxStageOutput {
        tensor: current,
        width: 1,
        kind: SKOnnxStageKind::Regression,
    })
}

/// Append the decision function and the link operator of a logistic model.
fn append_logistic_regression<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    let (shape, coefficients) = floating_tensor(state, "logistic.coefficients")?;
    let (_, intercepts) = floating_tensor(state, "logistic.intercepts")?;
    let &[heads, feature_count] = shape.as_slice() else {
        return Err(SKError::ShapeMismatch {
            expected: vec![1, state.metadata().feature_count()],
            found: shape,
        });
    };
    let class_count = state
        .metadata()
        .class_labels()
        .map(<[String]>::len)
        .unwrap_or(heads);
    let strategy = string_parameter(state, "strategy", "binary");
    if strategy == "binary" && heads != 1 {
        return Err(SKError::ShapeMismatch {
            expected: vec![1, feature_count],
            found: shape,
        });
    }
    if strategy == "multinomial" && heads != class_count {
        return Err(SKError::ShapeMismatch {
            expected: vec![class_count, feature_count],
            found: shape,
        });
    }
    let mut transposed = vec![F::zero(); coefficients.len()];
    for head in 0..heads {
        for feature in 0..feature_count {
            transposed[feature * heads + head] = coefficients[head * feature_count + feature];
        }
    }
    builder.add_floating_initializer(
        format!("{namespace}logistic.coefficients"),
        vec![feature_count, heads],
        &transposed,
    );
    let product = format!("{namespace}logistic.product");
    builder.add_node(SKOnnxNode::new(
        "MatMul",
        vec![
            input.to_string(),
            format!("{namespace}logistic.coefficients"),
        ],
        vec![product.clone()],
    ));
    let mut scores = product;
    if boolean_parameter(state, "include_intercept", true) {
        builder.add_floating_initializer(
            format!("{namespace}logistic.intercepts"),
            vec![1, heads],
            &intercepts,
        );
        let shifted = format!("{namespace}logistic.scores");
        builder.add_node(SKOnnxNode::new(
            "Add",
            vec![scores, format!("{namespace}logistic.intercepts")],
            vec![shifted.clone()],
        ));
        scores = shifted;
    }
    let probabilities = format!("{namespace}probabilities");
    match strategy {
        "binary" => {
            let positive = format!("{namespace}logistic.positive");
            builder.add_node(SKOnnxNode::new(
                "Sigmoid",
                vec![scores],
                vec![positive.clone()],
            ));
            builder.add_floating_initializer(
                format!("{namespace}logistic.one"),
                Vec::new(),
                &[F::one()],
            );
            let negative = format!("{namespace}logistic.negative");
            builder.add_node(SKOnnxNode::new(
                "Sub",
                vec![format!("{namespace}logistic.one"), positive.clone()],
                vec![negative.clone()],
            ));
            builder.add_node(
                SKOnnxNode::new(
                    "Concat",
                    vec![negative, positive],
                    vec![probabilities.clone()],
                )
                .attribute_integer("axis", 1),
            );
        }
        "multinomial" => {
            builder.add_node(
                SKOnnxNode::new("Softmax", vec![scores], vec![probabilities.clone()])
                    .attribute_integer("axis", 1),
            );
        }
        "one_versus_rest" => {
            let sigmoids = format!("{namespace}logistic.sigmoids");
            builder.add_node(SKOnnxNode::new(
                "Sigmoid",
                vec![scores],
                vec![sigmoids.clone()],
            ));
            let total = format!("{namespace}logistic.total");
            builder.add_node(
                SKOnnxNode::new("ReduceSum", vec![sigmoids.clone()], vec![total.clone()])
                    .attribute_integer("axis", 1)
                    .attribute_integer("keepdims", 1),
            );
            builder.add_node(SKOnnxNode::new(
                "Div",
                vec![sigmoids, total],
                vec![probabilities.clone()],
            ));
        }
        other => {
            return Err(unsupported(
                SK_LOGISTIC_REGRESSION_NAME,
                format!("the probability strategy `{other}` is not part of the exported subset"),
            ));
        }
    }
    let label_tensor = format!("{namespace}label");
    builder.add_node(
        SKOnnxNode::new(
            "ArgMax",
            vec![probabilities.clone()],
            vec![label_tensor.clone()],
        )
        .attribute_integer("axis", 1)
        .attribute_integer("keepdims", 0),
    );
    Ok(SKOnnxStageOutput {
        tensor: probabilities.clone(),
        width: class_count,
        kind: SKOnnxStageKind::Classification {
            label_tensor,
            probability_tensor: probabilities,
            class_count,
        },
    })
}
