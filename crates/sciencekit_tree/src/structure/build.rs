//! Structure-only tree growth over a pseudo-residual response vector.

use ndarray::ArrayView1;
use sciencekit_common::{SKDataView, SKError, SKFloat};

use crate::design::{SKDesign, sk_to_csc};
use crate::growth::{SKGrowthConfig, sk_grow_tree};
use crate::node::SKTree;

use super::constraints::SKStructureConstraints;
use super::criterion::{SKStructureCriterion, sk_structure_dimensions};

/// A grown CART split structure with placeholder leaf values, leaf-row
/// membership and normalized impurity-based feature importances.
#[derive(Debug, Clone)]
pub struct SKTreeStructure<F> {
    pub(super) tree: SKTree<F>,
    pub(super) leaf_node_indices: Vec<usize>,
    pub(super) leaf_sample_indices: Vec<Vec<usize>>,
    pub(super) feature_importances: Vec<F>,
}

impl<F: SKFloat> SKTreeStructure<F> {
    /// Grow a tree split structure on `pseudo_residuals` without committing to
    /// the public estimator's leaf-value policy (leaves start at `0.0`).
    pub fn build_structure<'a, X>(
        features: X,
        pseudo_residuals: ArrayView1<'_, F>,
        constraints: &SKStructureConstraints<F>,
    ) -> Result<Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let (rows_count, _) = sk_structure_dimensions(&data)?;
        if pseudo_residuals.len() != rows_count {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                1,
                pseudo_residuals.len(),
                1,
            ));
        }
        if let Some(position) = pseudo_residuals.iter().position(|value| !value.is_finite()) {
            return Err(SKError::NonFiniteValue { position });
        }
        let residual_values: Vec<F> = pseudo_residuals.iter().copied().collect();
        let configuration = SKGrowthConfig {
            maximum_depth: constraints.maximum_depth,
            minimum_samples_split: constraints.minimum_samples_split,
            minimum_samples_leaf: constraints.minimum_samples_leaf,
            maximum_features: constraints.maximum_features.clone(),
            splitter: constraints.splitter,
            random_seed: constraints.random_seed,
        };
        match data {
            SKDataView::Dense(dense) => {
                let design = SKDesign::Dense(dense);
                design.validate_finite()?;
                Ok(sk_build_from_design(
                    &design,
                    &configuration,
                    &residual_values,
                ))
            }
            SKDataView::Sparse(csr) => {
                let csc = sk_to_csc(&csr);
                let design = SKDesign::Sparse {
                    csr,
                    csc: csc.view(),
                };
                design.validate_finite()?;
                Ok(sk_build_from_design(
                    &design,
                    &configuration,
                    &residual_values,
                ))
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            }),
        }
    }

    /// Borrow the underlying flat [`SKTree`].
    pub fn tree(&self) -> &SKTree<F> {
        &self.tree
    }

    /// Consume the structure and return the finished [`SKTree`].
    pub fn into_tree(self) -> SKTree<F> {
        self.tree
    }

    /// The number of leaves in the grown structure.
    pub fn leaf_count(&self) -> usize {
        self.leaf_node_indices.len()
    }

    /// The flat node indices of every leaf, in ascending storage order.
    pub fn leaf_node_indices(&self) -> &[usize] {
        &self.leaf_node_indices
    }

    /// The training row indices (`0..rows`) that reached `leaf_ordinal`.
    pub fn leaf_sample_indices(&self, leaf_ordinal: usize) -> Option<&[usize]> {
        self.leaf_sample_indices
            .get(leaf_ordinal)
            .map(Vec::as_slice)
    }

    /// Normalized impurity-decrease feature importances of the split structure.
    pub fn feature_importances(&self) -> ArrayView1<'_, F> {
        ArrayView1::from(&self.feature_importances)
    }

    /// Total number of training samples used to grow this structure.
    pub fn sample_count(&self) -> usize {
        self.tree
            .node(self.tree.root_index())
            .map_or(0, |root| root.sample_count())
    }
}

fn sk_build_from_design<F: SKFloat>(
    design: &SKDesign<'_, F>,
    configuration: &SKGrowthConfig<F>,
    pseudo_residuals: &[F],
) -> SKTreeStructure<F> {
    let criterion = SKStructureCriterion { pseudo_residuals };
    let (tree, feature_importances) = sk_grow_tree(&criterion, design, configuration);
    let leaf_node_indices: Vec<usize> = tree
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(node_index, node)| node.is_leaf().then_some(node_index))
        .collect();
    let mut node_to_ordinal = vec![None; tree.node_count()];
    leaf_node_indices
        .iter()
        .enumerate()
        .for_each(|(ordinal, &node_index)| node_to_ordinal[node_index] = Some(ordinal));
    let mut leaf_sample_indices = vec![Vec::new(); leaf_node_indices.len()];
    (0..design.rows()).for_each(|row| {
        let leaf_node = tree.leaf_index_for(|feature| design.feature_value(row, feature));
        leaf_sample_indices[node_to_ordinal[leaf_node].expect("leaf mapped")].push(row);
    });
    SKTreeStructure {
        tree,
        leaf_node_indices,
        leaf_sample_indices,
        feature_importances,
    }
}
