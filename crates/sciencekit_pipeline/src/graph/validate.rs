//! DAG structural validation: edge bounds, cycle detection with cycle-path
//! naming, single final estimator sink and reachability (spec
//! `pipeline-graph`).

use std::collections::{BTreeSet, VecDeque};

use petgraph::Direction;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use sciencekit_common::{SKError, SKFloat};

use super::node::{GraphNodePayload, SKNodeIdentifier};

/// Validate a configured graph and return the unique final estimator node
/// identifier on success.
pub(super) fn sk_validate_graph<F: SKFloat>(
    graph: &DiGraph<GraphNodePayload<F>, usize>,
    invalid_endpoints: &[(SKNodeIdentifier, SKNodeIdentifier)],
) -> Result<SKNodeIdentifier, SKError> {
    if let Some(&(source, target)) = invalid_endpoints.first() {
        return Err(SKError::InvalidHyperparameter {
            name: "edges",
            reason: format!(
                "edge ({source} -> {target}) references a node outside 0..{}",
                graph.node_count()
            ),
        });
    }

    if toposort(graph, None).is_err() {
        let cycle_description = describe_cycle(graph);
        return Err(SKError::InvalidHyperparameter {
            name: "edges",
            reason: format!("pipeline graph contains a cycle: {cycle_description}"),
        });
    }

    let final_nodes: Vec<SKNodeIdentifier> = graph
        .node_indices()
        .filter(|&node| matches!(graph[node], GraphNodePayload::FinalEstimator(_)))
        .map(|node| SKNodeIdentifier::new(node.index()))
        .collect();

    if final_nodes.len() != 1 {
        let detail = if final_nodes.is_empty() {
            String::new()
        } else {
            let names: Vec<String> = final_nodes.iter().map(ToString::to_string).collect();
            format!(" ({})", names.join(", "))
        };
        return Err(SKError::InvalidHyperparameter {
            name: "final_estimator",
            reason: format!(
                "pipeline graph requires exactly one final estimator node, found {}{detail}",
                final_nodes.len()
            ),
        });
    }

    let final_node = final_nodes[0];
    let final_index = NodeIndex::new(final_node.index());
    if graph
        .edges_directed(final_index, Direction::Outgoing)
        .next()
        .is_some()
    {
        return Err(SKError::InvalidHyperparameter {
            name: "final_estimator",
            reason: format!("final estimator {final_node} cannot have outgoing edges"),
        });
    }

    validate_reachability(graph, final_index)?;
    Ok(final_node)
}

fn validate_reachability<F: SKFloat>(
    graph: &DiGraph<GraphNodePayload<F>, usize>,
    final_index: NodeIndex,
) -> Result<(), SKError> {
    let input_roots: Vec<NodeIndex> = graph
        .node_indices()
        .filter(|&node| {
            graph
                .edges_directed(node, Direction::Incoming)
                .next()
                .is_none()
        })
        .collect();

    let reachable_from_inputs = breadth_first_reach(graph, &input_roots, Direction::Outgoing);
    let reaches_final = breadth_first_reach(graph, &[final_index], Direction::Incoming);

    let disconnected: Vec<String> = graph
        .node_indices()
        .filter(|node| {
            !reachable_from_inputs.contains(&node.index()) || !reaches_final.contains(&node.index())
        })
        .map(|node| SKNodeIdentifier::new(node.index()).to_string())
        .collect();

    if !disconnected.is_empty() {
        return Err(SKError::InvalidHyperparameter {
            name: "edges",
            reason: format!(
                "pipeline graph contains unreachable or disconnected nodes: {}",
                disconnected.join(", ")
            ),
        });
    }
    Ok(())
}

fn breadth_first_reach<F: SKFloat>(
    graph: &DiGraph<GraphNodePayload<F>, usize>,
    seeds: &[NodeIndex],
    direction: Direction,
) -> BTreeSet<usize> {
    let mut visited: BTreeSet<usize> = seeds.iter().map(|node| node.index()).collect();
    let mut queue: VecDeque<NodeIndex> = seeds.iter().copied().collect();
    while let Some(current) = queue.pop_front() {
        graph
            .neighbors_directed(current, direction)
            .for_each(|neighbor| {
                if visited.insert(neighbor.index()) {
                    queue.push_back(neighbor);
                }
            });
    }
    visited
}

fn describe_cycle<F: SKFloat>(graph: &DiGraph<GraphNodePayload<F>, usize>) -> String {
    let mut visited_globally = BTreeSet::new();
    graph
        .node_indices()
        .find_map(|start| {
            if visited_globally.contains(&start.index()) {
                return None;
            }
            let mut active_stack = Vec::new();
            depth_first_cycle(graph, start, &mut active_stack, &mut visited_globally).map(|cycle| {
                let labels: Vec<String> = cycle
                    .into_iter()
                    .map(|index| SKNodeIdentifier::new(index).to_string())
                    .collect();
                labels.join(" -> ")
            })
        })
        .unwrap_or_else(|| "cyclic dependency".to_string())
}

fn depth_first_cycle<F: SKFloat>(
    graph: &DiGraph<GraphNodePayload<F>, usize>,
    current: NodeIndex,
    active_stack: &mut Vec<usize>,
    visited_globally: &mut BTreeSet<usize>,
) -> Option<Vec<usize>> {
    active_stack.push(current.index());
    let mut sorted_neighbors: Vec<NodeIndex> = graph
        .neighbors_directed(current, Direction::Outgoing)
        .collect();
    sorted_neighbors.sort_by_key(|node| node.index());

    let found = sorted_neighbors.into_iter().find_map(|neighbor| {
        if let Some(position) = active_stack
            .iter()
            .position(|&item| item == neighbor.index())
        {
            let mut cycle: Vec<usize> = active_stack[position..].to_vec();
            cycle.push(neighbor.index());
            return Some(cycle);
        }
        if !visited_globally.contains(&neighbor.index()) {
            return depth_first_cycle(graph, neighbor, active_stack, visited_globally);
        }
        None
    });

    if found.is_none() {
        active_stack.pop();
        visited_globally.insert(current.index());
    }
    found
}
