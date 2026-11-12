//! Deterministic level-ordered topological sorting and per-node block storage
//! tests (spec `pipeline-graph`, Task 2.1).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKFeatureBlock};
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use super::{SKNodeBlockStore, SKNodeIdentifier, SKPipelineGraph};

fn scaler() -> SKStandardScaler {
    SKStandardScaler::build().build().unwrap()
}

fn minmax() -> SKMinMaxScaler<f64> {
    SKMinMaxScaler::<f64>::build().build().unwrap()
}

fn classifier() -> SKLogisticRegression<f64> {
    SKLogisticRegression::build().build().unwrap()
}

#[test]
fn topological_order_respects_dependencies_when_nodes_are_inserted_reverse() {
    let mut builder = SKPipelineGraph::start::<f64>();
    let final_sink = builder.add_classifier(classifier());
    let branch_two = builder.add_transformer(minmax());
    let branch_one = builder.add_transformer(scaler());
    let root_input = builder.add_transformer(scaler());

    // Connect branch_one (node 2) before branch_two (node 1) to prove same-level
    // ordering uses ascending SKNodeIdentifier (node 1 before node 2).
    builder
        .connect(root_input, branch_one)
        .connect(root_input, branch_two)
        .connect(branch_one, final_sink)
        .connect(branch_two, final_sink);

    let graph = builder.build().unwrap();
    assert_eq!(
        graph.topological_order(),
        &[root_input, branch_two, branch_one, final_sink]
    );
}

#[test]
fn same_level_nodes_execute_in_ascending_node_identifier_order_independent_of_edge_insertion() {
    let build_with_edge_order = |connect_left_first: bool| {
        let mut builder = SKPipelineGraph::start::<f64>();
        let root = builder.add_transformer(scaler());
        let left = builder.add_transformer(scaler());
        let middle = builder.add_transformer(minmax());
        let right = builder.add_transformer(scaler());
        let sink = builder.add_classifier(classifier());

        if connect_left_first {
            builder
                .connect(root, left)
                .connect(root, middle)
                .connect(root, right);
        } else {
            builder
                .connect(root, right)
                .connect(root, middle)
                .connect(root, left);
        }
        builder
            .connect(left, sink)
            .connect(middle, sink)
            .connect(right, sink);
        builder.build().unwrap().topological_order().to_vec()
    };

    let forward_order = build_with_edge_order(true);
    let reverse_order = build_with_edge_order(false);
    assert_eq!(forward_order, reverse_order);
    assert_eq!(
        forward_order
            .iter()
            .map(|identifier| identifier.index())
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
}

#[test]
fn per_node_block_store_keeps_blocks_sorted_by_node_identifier() {
    let mut store = SKNodeBlockStore::<f64>::new();
    let second = SKNodeIdentifier::new(2);
    let zeroth = SKNodeIdentifier::new(0);
    let first = SKNodeIdentifier::new(1);

    store.insert(second, SKFeatureBlock::from(array![[5.0_f64, 6.0]]));
    store.insert(zeroth, SKFeatureBlock::from(array![[1.0_f64, 2.0]]));
    store.insert(first, SKFeatureBlock::from(array![[3.0_f64, 4.0]]));

    assert_eq!(store.len(), 3);
    assert!(!store.is_empty());
    assert_eq!(store.get(first).unwrap().number_of_columns(), 2);
    let keys: Vec<SKNodeIdentifier> = store.as_map().keys().copied().collect();
    assert_eq!(keys, vec![zeroth, first, second]);
}
