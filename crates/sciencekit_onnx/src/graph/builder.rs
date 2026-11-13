//! The name-checked ONNX graph builder.
//!
//! The builder validates the graph and assembles the shared
//! [`serde_onnx::ir::Model`] used by both directions of interoperability: the
//! exporter encodes that model, the importer decodes the same type.

use std::collections::{BTreeMap, BTreeSet};

use sciencekit_common::{SKError, SKFloat};
use serde_onnx::ir::{Dim, Graph, Model, OpsetId, Tensor, TensorData};

use super::{
    SK_ONNX_DEFAULT_OPSET, SK_ONNX_IR_VERSION, SK_ONNX_ML_DOMAIN, SK_ONNX_ML_TREE_OPSET,
    SKOnnxModel, SKOnnxNode, SKOnnxTensorDType, SKOnnxValueInfo,
};

/// Builds one ONNX graph and serializes it as a `ModelProto`.
///
/// The builder validates the graph before serializing: input, initializer and
/// node-output names are unique, every node input is produced by something
/// earlier in the graph and every declared output exists.
#[derive(Debug, Clone)]
pub struct SKOnnxGraphBuilder {
    graph_name: String,
    inputs: Vec<SKOnnxValueInfo>,
    outputs: Vec<SKOnnxValueInfo>,
    nodes: Vec<SKOnnxNode>,
    initializers: Vec<SKOnnxTensorDefinition>,
    metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
struct SKOnnxTensorDefinition {
    name: String,
    data_type: SKOnnxTensorDType,
    dimensions: Vec<usize>,
    data: TensorData,
}

impl SKOnnxGraphBuilder {
    /// Start a graph with a stable name.
    pub fn new(graph_name: impl Into<String>) -> Self {
        SKOnnxGraphBuilder {
            graph_name: graph_name.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            initializers: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Declare a graph input.
    pub fn add_input(&mut self, value_info: SKOnnxValueInfo) -> &mut Self {
        self.inputs.push(value_info);
        self
    }

    /// Declare a graph output.
    pub fn add_output(&mut self, value_info: SKOnnxValueInfo) -> &mut Self {
        self.outputs.push(value_info);
        self
    }

    /// Append an operator node.
    pub fn add_node(&mut self, node: SKOnnxNode) -> &mut Self {
        self.nodes.push(node);
        self
    }

    /// Attach a model metadata entry.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Serialize a floating tensor as a graph initializer.
    pub fn add_floating_initializer<F: SKFloat>(
        &mut self,
        name: impl Into<String>,
        dimensions: Vec<usize>,
        values: &[F],
    ) -> &mut Self {
        let data_type = SKOnnxTensorDType::for_scalar::<F>();
        let data = match data_type {
            SKOnnxTensorDType::Float => TensorData::F32(
                values
                    .iter()
                    .map(|value| value.to_f32().unwrap_or(f32::NAN))
                    .collect(),
            ),
            _ => TensorData::F64(
                values
                    .iter()
                    .map(|value| value.to_f64().unwrap_or(f64::NAN))
                    .collect(),
            ),
        };
        self.push_initializer(name, data_type, dimensions, data);
        self
    }

    /// Serialize an integer tensor as a graph initializer.
    pub fn add_integer_initializer(
        &mut self,
        name: impl Into<String>,
        dimensions: Vec<usize>,
        values: &[i64],
    ) -> &mut Self {
        self.push_initializer(
            name,
            SKOnnxTensorDType::Int64,
            dimensions,
            TensorData::I64(values.to_vec()),
        );
        self
    }

    fn push_initializer(
        &mut self,
        name: impl Into<String>,
        data_type: SKOnnxTensorDType,
        dimensions: Vec<usize>,
        data: TensorData,
    ) {
        self.initializers.push(SKOnnxTensorDefinition {
            name: name.into(),
            data_type,
            dimensions,
            data,
        });
    }

    /// Validate the graph and serialize it to ONNX bytes.
    pub fn build(self) -> Result<SKOnnxModel, SKError> {
        if self.inputs.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "onnx_graph_inputs",
                reason: "an ONNX graph needs at least one input".to_string(),
            });
        }
        if self.outputs.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "onnx_graph_outputs",
                reason: "an ONNX graph needs at least one output".to_string(),
            });
        }
        let mut produced: BTreeSet<String> = self
            .inputs
            .iter()
            .map(|input| input.name().to_string())
            .collect();
        for initializer in &self.initializers {
            if !produced.insert(initializer.name.clone()) {
                return Err(SKError::InvalidHyperparameter {
                    name: "onnx_graph_names",
                    reason: format!(
                        "graph value `{}` is defined more than once",
                        initializer.name
                    ),
                });
            }
            let expected: usize = initializer.dimensions.iter().product();
            let found = initializer.data.len().unwrap_or(expected);
            if found != expected {
                return Err(SKError::ShapeMismatch {
                    expected: initializer.dimensions.clone(),
                    found: vec![found],
                });
            }
        }
        for node in &self.nodes {
            for input in node.inputs() {
                if input.is_empty() {
                    continue;
                }
                if !produced.contains(input) {
                    return Err(SKError::InvalidHyperparameter {
                        name: "onnx_node_input",
                        reason: format!(
                            "node `{}` reads `{input}` before any producer defines it",
                            node.op_type()
                        ),
                    });
                }
            }
            for output in node.outputs() {
                if !produced.insert(output.clone()) {
                    return Err(SKError::InvalidHyperparameter {
                        name: "onnx_graph_names",
                        reason: format!("graph value `{output}` is defined more than once"),
                    });
                }
            }
        }
        for output in &self.outputs {
            if !produced.contains(output.name()) {
                return Err(SKError::InvalidHyperparameter {
                    name: "onnx_graph_output",
                    reason: format!(
                        "the graph declares output `{}` but no node produces it",
                        output.name()
                    ),
                });
            }
        }
        let mut graph = Graph::new(self.graph_name);
        graph.inputs = self.inputs.iter().map(SKOnnxValueInfo::to_ir).collect();
        graph.outputs = self.outputs.iter().map(SKOnnxValueInfo::to_ir).collect();
        graph.nodes = self.nodes.iter().map(SKOnnxNode::to_ir).collect();
        graph.initializers = self
            .initializers
            .iter()
            .map(SKOnnxTensorDefinition::to_ir)
            .collect();
        let mut model = Model::new(
            graph,
            vec![
                OpsetId {
                    domain: String::new(),
                    version: SK_ONNX_DEFAULT_OPSET,
                },
                OpsetId {
                    domain: SK_ONNX_ML_DOMAIN.to_string(),
                    version: SK_ONNX_ML_TREE_OPSET,
                },
            ],
        );
        model.ir_version = SK_ONNX_IR_VERSION;
        model.producer_name = "sciencekit".to_string();
        model.producer_version = env!("CARGO_PKG_VERSION").to_string();
        model.model_version = 1;
        model.metadata = self
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let bytes = serde_onnx::proto::encode_model(&model);
        SKOnnxModel::from_parts(bytes, model)
    }
}

impl SKOnnxTensorDefinition {
    fn to_ir(&self) -> Tensor {
        let shape = self
            .dimensions
            .iter()
            .map(|dimension| Dim::Fixed(*dimension as i64))
            .collect();
        Tensor::new(
            self.name.clone(),
            self.data_type.to_element_type(),
            shape,
            self.data.clone(),
        )
    }
}
