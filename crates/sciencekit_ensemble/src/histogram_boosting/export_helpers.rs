//! Shared state encoding for the histogram-gradient-boosting models.
//!
//! A fitted booster carries two learned artifacts that must both survive a
//! round trip: the per-feature binning tables (quantile edges or observed
//! category codes plus the missing bin) and the histogram tree nodes, including
//! the numeric missing-routing direction and the categorical membership masks.

use bitvec::{order::Lsb0, vec::BitVec};
use sciencekit_common::{
    SKError, SKFloat, SKModelState, SKTensor, sk_read_boolean_tensor, sk_read_floating_tensor,
    sk_read_integer_tensor,
};

use super::binning::SKFeatureBinning;
use super::growth::{SKHistogramNode, SKHistogramTree};
use super::histogram::SKHistogramSplit;

/// Encode the per-feature binning tables.
pub(crate) fn export_binnings<F: SKFloat>(
    binnings: &[SKFeatureBinning<F>],
) -> Result<Vec<SKTensor<F>>, SKError> {
    let mut tensors = Vec::new();
    for (index, binning) in binnings.iter().enumerate() {
        let prefix = format!("hist_binning.{index}");
        if let Some(missing_bin) = binning.missing_bin() {
            tensors.push(SKTensor::integer(
                format!("{prefix}.missing_bin"),
                Vec::new(),
                vec![missing_bin as i64],
            )?);
            tensors.push(SKTensor::floating(
                format!("{prefix}.edges"),
                vec![binning.edges().len()],
                binning.edges().to_vec(),
            )?);
        } else {
            tensors.push(SKTensor::integer(
                format!("{prefix}.categories"),
                vec![binning.categories().len()],
                binning.categories().to_vec(),
            )?);
        }
    }
    Ok(tensors)
}

/// Rebuild the per-feature binning tables for `feature_count` features.
pub(crate) fn import_binnings<F: SKFloat>(
    state: &SKModelState<F>,
    feature_count: usize,
) -> Result<Vec<SKFeatureBinning<F>>, SKError> {
    let mut binnings = Vec::with_capacity(feature_count);
    for index in 0..feature_count {
        let prefix = format!("hist_binning.{index}");
        if state.tensor(&format!("{prefix}.missing_bin")).is_some() {
            let missing = sk_read_integer_tensor(state, &format!("{prefix}.missing_bin"))?;
            let missing_bin = missing.first().copied().unwrap_or(255).clamp(0, 255) as u8;
            let (_, edges) = sk_read_floating_tensor(state, &format!("{prefix}.edges"))?;
            binnings.push(SKFeatureBinning::from_exported_numeric(edges, missing_bin));
        } else {
            let categories = sk_read_integer_tensor(state, &format!("{prefix}.categories"))?;
            binnings.push(SKFeatureBinning::from_exported_categorical(categories));
        }
    }
    Ok(binnings)
}

/// Encode one histogram tree under a name prefix.
pub(crate) fn export_histogram_tree<F: SKFloat>(
    tree: &SKHistogramTree<F>,
    prefix: &str,
) -> Result<Vec<SKTensor<F>>, SKError> {
    let nodes = tree.nodes();
    let mut kinds = Vec::with_capacity(nodes.len());
    let mut features = Vec::with_capacity(nodes.len());
    let mut maximum_bins = Vec::with_capacity(nodes.len());
    let mut default_left = Vec::with_capacity(nodes.len());
    let mut unknown_left = Vec::with_capacity(nodes.len());
    let mut missing_bins = Vec::with_capacity(nodes.len());
    let mut left = Vec::with_capacity(nodes.len());
    let mut right = Vec::with_capacity(nodes.len());
    let mut values = Vec::with_capacity(nodes.len());
    let mut sample_counts = Vec::with_capacity(nodes.len());
    let mut left_bins = Vec::new();
    let mut left_bin_offsets = Vec::with_capacity(nodes.len() + 1);
    left_bin_offsets.push(0_i64);
    for node in nodes {
        left.push(node.left_child().map_or(-1, |child| child as i64));
        right.push(node.right_child().map_or(-1, |child| child as i64));
        values.push(node.value());
        sample_counts.push(node.sample_count() as i64);
        match node.split() {
            None => {
                kinds.push(0_i64);
                features.push(-1_i64);
                maximum_bins.push(-1_i64);
                default_left.push(false);
                unknown_left.push(false);
                missing_bins.push(-1_i64);
            }
            Some(SKHistogramSplit::Numeric {
                feature_index,
                maximum_bin,
                default_left: node_default_left,
                missing_bin,
            }) => {
                kinds.push(1_i64);
                features.push(*feature_index as i64);
                maximum_bins.push(*maximum_bin as i64);
                default_left.push(*node_default_left);
                unknown_left.push(false);
                missing_bins.push(*missing_bin as i64);
            }
            Some(SKHistogramSplit::Categorical {
                feature_index,
                left_bins: left_bins_mask,
                unknown_left: node_unknown_left,
            }) => {
                kinds.push(2_i64);
                features.push(*feature_index as i64);
                maximum_bins.push(-1_i64);
                default_left.push(false);
                unknown_left.push(*node_unknown_left);
                missing_bins.push(-1_i64);
                left_bins.extend(left_bins_mask.iter().map(|bit| *bit));
            }
        }
        left_bin_offsets.push(left_bins.len() as i64);
    }
    Ok(vec![
        SKTensor::integer(format!("{prefix}.nodes.kind"), vec![nodes.len()], kinds)?,
        SKTensor::integer(
            format!("{prefix}.nodes.feature"),
            vec![nodes.len()],
            features,
        )?,
        SKTensor::integer(
            format!("{prefix}.nodes.maximum_bin"),
            vec![nodes.len()],
            maximum_bins,
        )?,
        SKTensor::boolean(
            format!("{prefix}.nodes.default_left"),
            vec![nodes.len()],
            default_left,
        )?,
        SKTensor::boolean(
            format!("{prefix}.nodes.unknown_left"),
            vec![nodes.len()],
            unknown_left,
        )?,
        SKTensor::integer(
            format!("{prefix}.nodes.missing_bin"),
            vec![nodes.len()],
            missing_bins,
        )?,
        SKTensor::integer(format!("{prefix}.nodes.left"), vec![nodes.len()], left)?,
        SKTensor::integer(format!("{prefix}.nodes.right"), vec![nodes.len()], right)?,
        SKTensor::floating(format!("{prefix}.nodes.value"), vec![nodes.len()], values)?,
        SKTensor::integer(
            format!("{prefix}.nodes.sample_count"),
            vec![nodes.len()],
            sample_counts,
        )?,
        SKTensor::boolean(
            format!("{prefix}.nodes.left_bins"),
            vec![left_bins.len()],
            left_bins,
        )?,
        SKTensor::integer(
            format!("{prefix}.nodes.left_bin_offsets"),
            vec![left_bin_offsets.len()],
            left_bin_offsets,
        )?,
    ])
}

/// Rebuild one histogram tree from a state.
pub(crate) fn import_histogram_tree<F: SKFloat>(
    state: &SKModelState<F>,
    prefix: &str,
    feature_count: usize,
) -> Result<SKHistogramTree<F>, SKError> {
    let kinds = sk_read_integer_tensor(state, &format!("{prefix}.nodes.kind"))?;
    let features = sk_read_integer_tensor(state, &format!("{prefix}.nodes.feature"))?;
    let maximum_bins = sk_read_integer_tensor(state, &format!("{prefix}.nodes.maximum_bin"))?;
    let default_left = sk_read_boolean_tensor(state, &format!("{prefix}.nodes.default_left"))?;
    let unknown_left = sk_read_boolean_tensor(state, &format!("{prefix}.nodes.unknown_left"))?;
    let missing_bins = sk_read_integer_tensor(state, &format!("{prefix}.nodes.missing_bin"))?;
    let left = sk_read_integer_tensor(state, &format!("{prefix}.nodes.left"))?;
    let right = sk_read_integer_tensor(state, &format!("{prefix}.nodes.right"))?;
    let (_, values) = sk_read_floating_tensor(state, &format!("{prefix}.nodes.value"))?;
    let sample_counts = sk_read_integer_tensor(state, &format!("{prefix}.nodes.sample_count"))?;
    let left_bins = sk_read_boolean_tensor(state, &format!("{prefix}.nodes.left_bins"))?;
    let left_bin_offsets =
        sk_read_integer_tensor(state, &format!("{prefix}.nodes.left_bin_offsets"))?;
    let node_count = kinds.len();
    let consistent = features.len() == node_count
        && maximum_bins.len() == node_count
        && default_left.len() == node_count
        && unknown_left.len() == node_count
        && missing_bins.len() == node_count
        && left.len() == node_count
        && right.len() == node_count
        && values.len() == node_count
        && sample_counts.len() == node_count
        && left_bin_offsets.len() == node_count + 1;
    if !consistent {
        return Err(SKError::ShapeMismatch {
            expected: vec![node_count],
            found: vec![features.len(), values.len(), left_bin_offsets.len()],
        });
    }
    let mut nodes = Vec::with_capacity(node_count);
    for index in 0..node_count {
        let node_left = (left[index] >= 0).then_some(left[index] as usize);
        let node_right = (right[index] >= 0).then_some(right[index] as usize);
        let split = match kinds[index] {
            0 => None,
            1 => {
                if features[index] < 0 || features[index] as usize >= feature_count {
                    return Err(SKError::ShapeMismatch {
                        expected: vec![feature_count],
                        found: vec![features[index].max(0) as usize],
                    });
                }
                Some(SKHistogramSplit::Numeric {
                    feature_index: features[index] as usize,
                    maximum_bin: maximum_bins[index].clamp(0, 255) as u8,
                    default_left: default_left[index],
                    missing_bin: missing_bins[index].clamp(0, 255) as u8,
                })
            }
            2 => {
                if features[index] < 0 || features[index] as usize >= feature_count {
                    return Err(SKError::ShapeMismatch {
                        expected: vec![feature_count],
                        found: vec![features[index].max(0) as usize],
                    });
                }
                let start = left_bin_offsets[index].max(0) as usize;
                let end = left_bin_offsets[index + 1].max(0) as usize;
                let mask_bits = left_bins.get(start..end).ok_or(SKError::ShapeMismatch {
                    expected: vec![left_bins.len()],
                    found: vec![end],
                })?;
                let mask: BitVec<u8, Lsb0> = mask_bits.iter().copied().collect();
                Some(SKHistogramSplit::Categorical {
                    feature_index: features[index] as usize,
                    left_bins: mask,
                    unknown_left: unknown_left[index],
                })
            }
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown histogram node kind",
                    suggestion: "export a leaf, numeric or categorical node",
                });
            }
        };
        nodes.push(SKHistogramNode::from_exported(
            split,
            node_left,
            node_right,
            values[index],
            sample_counts[index].max(0) as usize,
        ));
    }
    Ok(SKHistogramTree::from_exported(nodes, feature_count))
}
