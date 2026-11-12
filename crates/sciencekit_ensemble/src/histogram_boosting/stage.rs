//! One boosting stage: loss derivatives, row subsampling, histogram growth and
//! the shrunk raw-score update (spec `histogram-gradient-boosting`, design
//! decision 7).

use sciencekit_common::{SKError, SKFloat};

use crate::boosting::loss::SKGradientLoss;
use crate::boosting::subsample::sk_select_stage_rows;
use crate::histogram_boosting::binning::SKBinnedDataset;
use crate::histogram_boosting::growth::{
    SKHistogramGrowthParameters, SKHistogramGrowthResult, SKHistogramTree, sk_grow_histogram_tree,
};

/// Per-stage controls of the histogram booster.
#[derive(Debug, Clone)]
pub(crate) struct SKHistogramStageParameters<F> {
    pub(crate) growth: SKHistogramGrowthParameters<F>,
    pub(crate) learning_rate: F,
    pub(crate) subsample: F,
    pub(crate) random_seed: u64,
}

/// Fit one scalar histogram tree on the loss derivatives and add its shrunk
/// leaf values into `raw_scores`.
pub(crate) fn sk_fit_histogram_scalar_stage<F: SKFloat, L: SKGradientLoss<F>>(
    loss: &L,
    dataset: &SKBinnedDataset<F>,
    targets: &[F],
    raw_scores: &mut [F],
    parameters: &SKHistogramStageParameters<F>,
    stage_index: usize,
    training_rows: &[usize],
) -> Result<SKHistogramGrowthResult<F>, SKError> {
    let negative_gradients: Vec<F> = targets
        .iter()
        .zip(raw_scores.iter())
        .map(|(&target, &score)| loss.negative_gradient(target, score))
        .collect();
    let hessians: Vec<F> = targets
        .iter()
        .zip(raw_scores.iter())
        .map(|(&target, &score)| loss.hessian(target, score))
        .collect();
    let rows = sk_select_stage_rows(
        training_rows.len(),
        parameters.subsample,
        parameters.random_seed,
        stage_index,
        0,
    )
    .into_iter()
    .map(|position| training_rows[position])
    .collect::<Vec<usize>>();
    let result = sk_grow_histogram_tree(
        dataset,
        &negative_gradients,
        &hessians,
        &rows,
        &parameters.growth,
    );
    sk_apply_tree_to_scores(
        dataset,
        &result.tree,
        parameters.learning_rate,
        raw_scores,
    );
    Ok(result)
}

/// Add `learning_rate * tree(binned row)` into every raw score.
pub(crate) fn sk_apply_tree_to_scores<F: SKFloat>(
    dataset: &SKBinnedDataset<F>,
    tree: &SKHistogramTree<F>,
    learning_rate: F,
    raw_scores: &mut [F],
) {
    raw_scores
        .iter_mut()
        .enumerate()
        .for_each(|(row, raw_score)| {
            let step = tree.leaf_value_for(|feature| dataset.bin(feature, row));
            *raw_score = *raw_score + learning_rate * step;
        });
}
