//! One-vs-one aggregation for a multiclass SVC (votes plus scaled margin sums).

use sciencekit_common::{SKError, SKFloat};

use crate::graph::{SKOnnxGraphBuilder, SKOnnxNode};

/// Concatenate `[-m, +m]` so `ArgMax` reproduces the Rust vote (zero is negative).
pub(crate) fn concat_polarity(
    builder: &mut SKOnnxGraphBuilder,
    margin: &str,
    namespace: &str,
) -> String {
    let negated = format!("{namespace}svm.negated_margin");
    builder.add_node(SKOnnxNode::new(
        "Neg",
        vec![margin.to_string()],
        vec![negated.clone()],
    ));
    let doubled = format!("{namespace}svm.polarity");
    builder.add_node(
        SKOnnxNode::new(
            "Concat",
            vec![negated, margin.to_string()],
            vec![doubled.clone()],
        )
        .attribute_integer("axis", 1),
    );
    doubled
}

/// Aggregate pairwise margins into `(votes, confidences)` per class.
///
/// `votes_c = Σ_p 1[class ]`, `margin_sum_c = Σ_p ±m_p`; the confidence is
/// `votes + margin_sum / (3 · (|margin_sum| + 1))`, exactly as the Rust
/// `sk_aggregate_one_vs_one_confidences`.
pub(crate) fn append_one_vs_one<F: SKFloat>(
    builder: &mut SKOnnxGraphBuilder,
    margins: &[String],
    pairs: &[(usize, usize)],
    classes: usize,
    namespace: &str,
) -> Result<(String, String), SKError> {
    builder.add_floating_initializer(format!("{namespace}svm.one"), Vec::new(), &[F::one()]);
    builder.add_floating_initializer(format!("{namespace}svm.zero"), Vec::new(), &[F::zero()]);
    builder.add_floating_initializer(
        format!("{namespace}svm.three"),
        Vec::new(),
        &[F::from(3.0).ok_or(SKError::Conversion(
            "3.0 does not fit the scalar".to_string(),
        ))?],
    );
    let mut votes: Vec<Option<String>> = vec![None; classes];
    let mut margin_sums: Vec<Option<String>> = vec![None; classes];
    for (index, (negative, positive)) in pairs.iter().enumerate() {
        let winner = format!("{namespace}svm.winner.{index}");
        builder.add_node(SKOnnxNode::new(
            "Greater",
            vec![margins[index].clone(), format!("{namespace}svm.zero")],
            vec![winner.clone()],
        ));
        let winner_value = format!("{namespace}svm.winner_value.{index}");
        builder.add_node(
            SKOnnxNode::new("Cast", vec![winner], vec![winner_value.clone()])
                .attribute_integer("to", super::super::helpers::element_type::<F>()),
        );
        let loser = format!("{namespace}svm.loser.{index}");
        builder.add_node(SKOnnxNode::new(
            "Sub",
            vec![format!("{namespace}svm.one"), winner_value.clone()],
            vec![loser.clone()],
        ));
        for (class, contribution) in [(*negative, loser), (*positive, winner_value)] {
            votes[class] = Some(accumulate(
                builder,
                votes[class].take(),
                contribution,
                &format!("{namespace}svm.vote_sum.{index}.{class}"),
            ));
        }
        let negated = format!("{namespace}svm.margin_negated.{index}");
        builder.add_node(SKOnnxNode::new(
            "Neg",
            vec![margins[index].clone()],
            vec![negated.clone()],
        ));
        for (class, contribution) in [(*negative, negated), (*positive, margins[index].clone())] {
            margin_sums[class] = Some(accumulate(
                builder,
                margin_sums[class].take(),
                contribution,
                &format!("{namespace}svm.margin_sum.{index}.{class}"),
            ));
        }
    }
    let vote_columns: Vec<String> = votes
        .into_iter()
        .map(|column| column.unwrap_or_else(|| format!("{namespace}svm.zero")))
        .collect();
    let votes_all = format!("{namespace}svm.votes");
    builder.add_node(
        SKOnnxNode::new("Concat", vote_columns, vec![votes_all.clone()])
            .attribute_integer("axis", 1),
    );
    let mut confidences = Vec::with_capacity(classes);
    for (class, margin_sum) in margin_sums.into_iter().enumerate() {
        let margin_sum = margin_sum.unwrap_or_else(|| format!("{namespace}svm.zero"));
        let absolute = format!("{namespace}svm.margin_abs.{class}");
        builder.add_node(SKOnnxNode::new(
            "Abs",
            vec![margin_sum.clone()],
            vec![absolute.clone()],
        ));
        let shifted = format!("{namespace}svm.margin_shift.{class}");
        builder.add_node(SKOnnxNode::new(
            "Add",
            vec![absolute, format!("{namespace}svm.one")],
            vec![shifted.clone()],
        ));
        let denominator = format!("{namespace}svm.margin_denominator.{class}");
        builder.add_node(SKOnnxNode::new(
            "Mul",
            vec![shifted, format!("{namespace}svm.three")],
            vec![denominator.clone()],
        ));
        let scaled = format!("{namespace}svm.margin_scaled.{class}");
        builder.add_node(SKOnnxNode::new(
            "Div",
            vec![margin_sum, denominator],
            vec![scaled.clone()],
        ));
        let vote = format!("{namespace}svm.vote_column.{class}");
        builder.add_node(
            SKOnnxNode::new(
                "Gather",
                vec![
                    votes_all.clone(),
                    format!("{namespace}svm.class_index.{class}"),
                ],
                vec![vote.clone()],
            )
            .attribute_integer("axis", 1),
        );
        builder.add_integer_initializer(
            format!("{namespace}svm.class_index.{class}"),
            vec![1],
            &[class as i64],
        );
        let confidence = format!("{namespace}svm.confidence.{class}");
        builder.add_node(SKOnnxNode::new(
            "Add",
            vec![vote, scaled],
            vec![confidence.clone()],
        ));
        confidences.push(confidence);
    }
    let decision = format!("{namespace}decision_values");
    builder.add_node(
        SKOnnxNode::new("Concat", confidences, vec![decision.clone()]).attribute_integer("axis", 1),
    );
    Ok((votes_all, decision))
}

fn accumulate(
    builder: &mut SKOnnxGraphBuilder,
    previous: Option<String>,
    contribution: String,
    name: &str,
) -> String {
    match previous {
        None => contribution,
        Some(previous) => {
            builder.add_node(SKOnnxNode::new(
                "Add",
                vec![previous, contribution],
                vec![name.to_string()],
            ));
            name.to_string()
        }
    }
}
