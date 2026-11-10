//! Best-first traversal and per-row search for the ball tree.

use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use sciencekit_common::SKFloat;

use super::super::build::SKBallTree;
use crate::metric::sk_lower_bound_exceeds;
use crate::searcher::candidates::{
    SKCandidate, sk_current_worst, sk_offer_candidate, sk_sort_candidates,
};
use crate::searcher::{SKNeighborQuery, SKRadiusNeighbors};

impl<F: SKFloat> SKBallTree<F> {
    /// Evaluate a query batch with an explicit parallel/sequential choice.
    pub(crate) fn k_nearest_with_mode(
        &self,
        queries: ArrayView2<'_, F>,
        neighbors_count: usize,
        parallel: bool,
    ) -> SKNeighborQuery<F> {
        let rows = queries.nrows();
        let per_row = |row: usize| self.k_nearest_row(queries.row(row), neighbors_count);
        let results: Vec<Vec<SKCandidate<F>>> = if parallel {
            (0..rows).into_par_iter().map(per_row).collect()
        } else {
            (0..rows).map(per_row).collect()
        };

        let mut distances = Array2::from_elem((rows, neighbors_count), F::zero());
        let mut indices = Array2::zeros((rows, neighbors_count));
        for (row, best) in results.into_iter().enumerate() {
            for (column, candidate) in best.into_iter().enumerate() {
                distances[[row, column]] = candidate.distance;
                indices[[row, column]] = candidate.index;
            }
        }
        SKNeighborQuery::new(distances, indices)
    }

    /// Evaluate a radius batch with an explicit parallel/sequential choice.
    pub(crate) fn radius_neighbors_with_mode(
        &self,
        queries: ArrayView2<'_, F>,
        radius: F,
        parallel: bool,
    ) -> SKRadiusNeighbors<F> {
        let rows = queries.nrows();
        let per_row = |row: usize| self.radius_neighbors_row(queries.row(row), radius);
        let results: Vec<Vec<SKCandidate<F>>> = if parallel {
            (0..rows).into_par_iter().map(per_row).collect()
        } else {
            (0..rows).map(per_row).collect()
        };

        let mut distances = Vec::with_capacity(rows);
        let mut indices = Vec::with_capacity(rows);
        for mut candidates in results {
            sk_sort_candidates(&mut candidates);
            distances.push(candidates.iter().map(|c| c.distance).collect());
            indices.push(candidates.iter().map(|c| c.index).collect());
        }
        SKRadiusNeighbors::new(distances, indices)
    }

    fn k_nearest_row(
        &self,
        query: ArrayView1<'_, F>,
        neighbors_count: usize,
    ) -> Vec<SKCandidate<F>> {
        let mut best: Vec<SKCandidate<F>> = Vec::with_capacity(neighbors_count);
        let mut stack: Vec<(usize, F)> = Vec::new();
        self.push_root(&mut stack, query);

        while let Some((node_index, lower_bound)) = stack.pop() {
            let worst = sk_current_worst(&best, neighbors_count);
            if sk_lower_bound_exceeds(lower_bound, worst) {
                continue;
            }
            let node = &self.nodes()[node_index];
            if node.is_leaf() {
                for &row in &self.permutation()[node.start()..node.end()] {
                    let distance = self
                        .metric
                        .point_distance(query, self.training_features.row(row));
                    sk_offer_candidate(&mut best, distance, row, neighbors_count);
                }
            } else {
                self.push_children(&mut stack, node_index, query);
            }
        }
        best
    }

    fn radius_neighbors_row(&self, query: ArrayView1<'_, F>, radius: F) -> Vec<SKCandidate<F>> {
        let mut found: Vec<SKCandidate<F>> = Vec::new();
        let mut stack: Vec<(usize, F)> = Vec::new();
        self.push_root(&mut stack, query);

        while let Some((node_index, lower_bound)) = stack.pop() {
            if sk_lower_bound_exceeds(lower_bound, radius) {
                continue;
            }
            let node = &self.nodes()[node_index];
            if node.is_leaf() {
                for &row in &self.permutation()[node.start()..node.end()] {
                    let distance = self
                        .metric
                        .point_distance(query, self.training_features.row(row));
                    if distance <= radius {
                        found.push(SKCandidate {
                            distance,
                            index: row,
                        });
                    }
                }
            } else {
                self.push_children(&mut stack, node_index, query);
            }
        }
        sk_sort_candidates(&mut found);
        found
    }

    fn push_root(&self, stack: &mut Vec<(usize, F)>, query: ArrayView1<'_, F>) {
        let root = &self.nodes()[0];
        let bound = self
            .metric
            .ball_lower_bound(query, root.centroid(), root.radius());
        stack.push((0, bound));
    }

    fn push_children(
        &self,
        stack: &mut Vec<(usize, F)>,
        node_index: usize,
        query: ArrayView1<'_, F>,
    ) {
        let node = &self.nodes()[node_index];
        let (left, right) = (
            node.left().expect("internal node"),
            node.right().expect("internal node"),
        );
        let left_node = &self.nodes()[left];
        let right_node = &self.nodes()[right];
        let left_bound =
            self.metric
                .ball_lower_bound(query, left_node.centroid(), left_node.radius());
        let right_bound =
            self.metric
                .ball_lower_bound(query, right_node.centroid(), right_node.radius());
        if left_bound <= right_bound {
            stack.push((right, right_bound));
            stack.push((left, left_bound));
        } else {
            stack.push((left, left_bound));
            stack.push((right, right_bound));
        }
    }
}
