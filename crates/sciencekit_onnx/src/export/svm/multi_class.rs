//! `SKSVC` export: pairwise binary heads plus one-vs-one aggregation.

use sciencekit_common::{SKError, SKFloat, SKModelState};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

use super::super::helpers::{floating_tensor, integer_parameter, unsupported};
use super::super::stage::{SKOnnxStageKind, SKOnnxStageOutput};
use super::classification::{append_one_vs_one, concat_polarity};
use super::dispatcher::SK_SVC_NAME;
use super::expansion::{append_expansion, dense_support_vectors, read_bias};
use super::kernel::require_exportable_kernel;

pub(crate) fn append_svc<F: SKFloat>(
    state: &SKModelState<F>,
    builder: &mut SKOnnxGraphBuilder,
    input: &str,
    namespace: &str,
) -> Result<SKOnnxStageOutput, SKError> {
    require_exportable_kernel(state)?;
    let pair_count = integer_parameter(state, "number_of_pairs", 0).max(0) as usize;
    if pair_count == 0 {
        return Err(unsupported(
            SK_SVC_NAME,
            "the state carries no pairwise head",
        ));
    }
    let mut margins = Vec::with_capacity(pair_count);
    let mut pairs = Vec::with_capacity(pair_count);
    for index in 0..pair_count {
        let prefix = format!("svc.pair.{index}");
        let (shape, support_vectors) = dense_support_vectors(state, &prefix, SK_SVC_NAME)?;
        builder.add_floating_initializer(
            format!("{namespace}svc.pair.{index}.support_vectors"),
            shape,
            &support_vectors,
        );
        let (_, dual_coefficients) =
            floating_tensor(state, &format!("{prefix}.dual_coefficients"))?;
        let bias = read_bias(state, &prefix)?;
        let margin = append_expansion(
            state,
            builder,
            input,
            &format!("{namespace}svc.pair.{index}.support_vectors"),
            &dual_coefficients,
            bias,
            &format!("{namespace}svc.pair.{index}."),
        )?;
        margins.push(margin);
        pairs.push((
            integer_parameter(state, &format!("{prefix}.class_negative"), 0).max(0) as usize,
            integer_parameter(state, &format!("{prefix}.class_positive"), 1).max(0) as usize,
        ));
    }
    let classes = state
        .metadata()
        .class_labels()
        .map(<[String]>::len)
        .unwrap_or_else(|| {
            pairs
                .iter()
                .map(|(_, positive)| positive + 1)
                .max()
                .unwrap_or(2)
        });
    if pair_count == 1 {
        let margin = margins.pop().expect("one pair");
        let polarity = concat_polarity(builder, &margin, namespace);
        let label = format!("{namespace}label");
        builder.add_node(
            SKOnnxNode::new("ArgMax", vec![polarity], vec![label.clone()])
                .attribute_integer("axis", 1)
                .attribute_integer("keepdims", 0),
        );
        return Ok(SKOnnxStageOutput {
            tensor: margin.clone(),
            width: 1,
            kind: SKOnnxStageKind::DecisionValues {
                label_tensor: label,
                decision_tensor: margin,
                decision_width: 1,
            },
        });
    }
    let (votes, decision) = append_one_vs_one::<F>(builder, &margins, &pairs, classes, namespace)?;
    let label = format!("{namespace}label");
    builder.add_node(
        SKOnnxNode::new("ArgMax", vec![votes], vec![label.clone()])
            .attribute_integer("axis", 1)
            .attribute_integer("keepdims", 0),
    );
    Ok(SKOnnxStageOutput {
        tensor: decision.clone(),
        width: classes,
        kind: SKOnnxStageKind::DecisionValues {
            label_tensor: label,
            decision_tensor: decision,
            decision_width: classes,
        },
    })
}
