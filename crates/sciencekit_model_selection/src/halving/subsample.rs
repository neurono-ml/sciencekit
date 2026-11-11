//! Deterministic per-round resource subsampling
//! (spec `halving-grid-search`, requirement `Resource subsampling`).

use std::collections::BTreeMap;

use ndarray::{Array1, ArrayView1};
use sciencekit_common::{
    SK_RANDOM_DOMAIN_FOLD, SK_RANDOM_DOMAIN_STAGE, SKDataView, SKError, SKFeatureBlock, SKFloat,
    SKRandomGenerator, SKTargetView, sk_derive_seed,
};

use crate::grid_search::observation::sk_view_dimensions;
use crate::split::validation::sk_class_groups;
use crate::train_test_split::sk_allocate_class_counts;
use crate::{SKTargetPartition, sk_gather_rows};

/// Owned subset of features, targets and optional groups gathered for a single round.
pub(crate) struct SKRoundSubsample<F: SKFloat> {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) row_indices: Vec<usize>,
    pub(crate) features: SKFeatureBlock<F>,
    pub(crate) targets: SKTargetPartition<F>,
    pub(crate) groups: Option<Array1<i64>>,
}

/// Selects deterministic row indices for a halving round, preserving class proportions
/// when `stratify` is enabled or group balance when `groups` are supplied.
pub(crate) fn sk_select_subsample_indices<F: SKFloat>(
    total_rows: usize,
    resource: usize,
    random_seed: u64,
    round_index: usize,
    stratify: bool,
    targets: SKTargetView<'_, F>,
    groups: Option<ArrayView1<'_, i64>>,
) -> Result<Vec<usize>, SKError> {
    if total_rows == 0 || resource == 0 {
        return Err(SKError::EmptyInput {
            operation: "halving_subsample",
        });
    }
    if resource >= total_rows {
        return Ok((0..total_rows).collect());
    }

    let round_seed = sk_derive_seed(random_seed, SK_RANDOM_DOMAIN_STAGE, round_index as u64);
    if stratify {
        let class_groups = sk_class_groups(Some(targets))?;
        return Ok(sk_draw_proportional_groups(
            &class_groups,
            resource,
            total_rows,
            round_seed,
        ));
    }

    if let Some(group_values) = groups {
        let grouped = sk_group_member_indices(group_values);
        return Ok(sk_draw_proportional_groups(
            &grouped, resource, total_rows, round_seed,
        ));
    }

    let mut shuffled = SKRandomGenerator::new(round_seed).shuffled_indices(total_rows);
    shuffled.truncate(resource);
    Ok(shuffled)
}

/// Gathers the subsampled features, targets and optional groups for a halving round
/// without densifying sparse CSR inputs.
pub(crate) fn sk_gather_round_subsample<F: SKFloat>(
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
    groups: Option<ArrayView1<'_, i64>>,
    resource: usize,
    random_seed: u64,
    round_index: usize,
    stratify: bool,
) -> Result<SKRoundSubsample<F>, SKError> {
    let (total_rows, _) = sk_view_dimensions(features);
    if let Some(group_values) = groups {
        if group_values.len() != total_rows {
            return Err(SKError::ShapeMismatch {
                expected: vec![total_rows],
                found: vec![group_values.len()],
            });
        }
    }

    let row_indices = sk_select_subsample_indices(
        total_rows,
        resource,
        random_seed,
        round_index,
        stratify,
        targets,
        groups,
    )?;
    let (gathered_features, gathered_targets) =
        sk_gather_rows(features, Some(targets), &row_indices)?;
    let gathered_targets = gathered_targets.expect("targets were supplied to sk_gather_rows");
    let gathered_groups = groups.map(|group_values| {
        row_indices
            .iter()
            .map(|&row| group_values[row])
            .collect::<Array1<i64>>()
    });

    Ok(SKRoundSubsample {
        row_indices,
        features: gathered_features,
        targets: gathered_targets,
        groups: gathered_groups,
    })
}

fn sk_draw_proportional_groups(
    member_groups: &[Vec<usize>],
    resource: usize,
    total_rows: usize,
    round_seed: u64,
) -> Vec<usize> {
    let group_sizes: Vec<usize> = member_groups.iter().map(Vec::len).collect();
    let allocated = sk_allocate_class_counts(&group_sizes, resource, total_rows);

    let mut selected: Vec<usize> = member_groups
        .iter()
        .zip(allocated)
        .enumerate()
        .flat_map(|(group_index, (members, count))| {
            let group_seed = sk_derive_seed(round_seed, SK_RANDOM_DOMAIN_FOLD, group_index as u64);
            let mut ordered = members.clone();
            SKRandomGenerator::new(group_seed).shuffle(&mut ordered);
            ordered.truncate(count);
            ordered
        })
        .collect();

    SKRandomGenerator::new(round_seed).shuffle(&mut selected);
    selected
}

fn sk_group_member_indices(groups: ArrayView1<'_, i64>) -> Vec<Vec<usize>> {
    let mut groups_by_value: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    groups
        .iter()
        .enumerate()
        .for_each(|(row, &group)| groups_by_value.entry(group).or_default().push(row));
    groups_by_value.into_values().collect()
}
