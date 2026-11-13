//! The name-checked ONNX graph builder.

use std::collections::{BTreeMap, BTreeSet};

use sciencekit_common::{SKError, SKFloat};

use super::{
    SK_ONNX_DEFAULT_OPSET, SK_ONNX_IR_VERSION, SK_ONNX_ML_DOMAIN, SK_ONNX_ML_TREE_OPSET,
    SKOnnxModel, SKOnnxNode, SKOnnxTensorDType, SKOnnxValueInfo, proto,
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
    raw_data: Vec<u8>,
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
        let raw_data = match data_type {
            SKOnnxTensorDType::Float => values
                .iter()
                .flat_map(|value| value.to_f32().unwrap_or(f32::NAN).to_le_bytes())
                .collect(),
            _ => values
                .iter()
                .flat_map(|value| value.to_f64().unwrap_or(f64::NAN).to_le_bytes())
                .collect(),
        };
        self.push_initializer(name, data_type, dimensions, raw_data);
        self
    }

    /// Serialize an integer tensor as a graph initializer.
    pub fn add_integer_initializer(
        &mut self,
        name: impl Into<String>,
        dimensions: Vec<usize>,
        values: &[i64],
    ) -> &mut Self {
        let raw_data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        self.push_initializer(name, SKOnnxTensorDType::Int64, dimensions, raw_data);
        self
    }

    fn push_initializer(
        &mut self,
        name: impl Into<String>,
        data_type: SKOnnxTensorDType,
        dimensions: Vec<usize>,
        raw_data: Vec<u8>,
    ) {
        self.initializers.push(SKOnnxTensorDefinition {
            name: name.into(),
            data_type,
            dimensions,
            raw_data,
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
            let bytes_per_value = bytes_per_value(initializer.data_type);
            if initializer.raw_data.len() != expected * bytes_per_value {
                return Err(SKError::ShapeMismatch {
                    expected: initializer.dimensions.clone(),
                    found: vec![initializer.raw_data.len() / bytes_per_value.max(1)],
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
        let mut opset_import = vec![
            proto::SkOnnxOperatorSetIdProto {
                domain: String::new(),
                version: SK_ONNX_DEFAULT_OPSET,
            },
            proto::SkOnnxOperatorSetIdProto {
                domain: SK_ONNX_ML_DOMAIN.to_string(),
                version: SK_ONNX_ML_TREE_OPSET,
            },
        ];
        opset_import.sort_by(|left, right| left.domain.cmp(&right.domain));
        let model_proto = proto::SkOnnxModelProto {
            ir_version: SK_ONNX_IR_VERSION,
            producer_name: "sciencekit".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            graph: Some(proto::SkOnnxGraphProto {
                node: self.nodes.iter().map(SKOnnxNode::to_proto).collect(),
                name: self.graph_name.clone(),
                initializer: self.initializers.iter().map(initializer_to_proto).collect(),
                doc_string: String::new(),
                input: self.inputs.iter().map(SKOnnxValueInfo::to_proto).collect(),
                output: self.outputs.iter().map(SKOnnxValueInfo::to_proto).collect(),
            }),
            opset_import,
            metadata_props: self
                .metadata
                .iter()
                .map(|(key, value)| proto::SkOnnxStringStringEntryProto {
                    key: key.clone(),
                    value: value.clone(),
                })
                .collect(),
        };
        let bytes = prost::Message::encode_to_vec(&model_proto);
        let node_count = self.nodes.len();
        let initializer_count = self.initializers.len();
        Ok(SKOnnxModel::new(
            bytes,
            SK_ONNX_DEFAULT_OPSET,
            SK_ONNX_ML_TREE_OPSET,
            self.inputs,
            self.outputs,
            self.metadata,
            node_count,
            initializer_count,
        ))
    }
}

fn bytes_per_value(data_type: SKOnnxTensorDType) -> usize {
    match data_type {
        SKOnnxTensorDType::Float => 4,
        SKOnnxTensorDType::Double => 8,
        SKOnnxTensorDType::Int64 => 8,
        SKOnnxTensorDType::Boolean => 1,
    }
}

fn initializer_to_proto(initializer: &SKOnnxTensorDefinition) -> proto::SkOnnxTensorProto {
    proto::SkOnnxTensorProto {
        dims: initializer
            .dimensions
            .iter()
            .map(|dimension| *dimension as i64)
            .collect(),
        data_type: initializer.data_type.element_type(),
        name: initializer.name.clone(),
        raw_data: initializer.raw_data.clone(),
        doc_string: String::new(),
    }
}
