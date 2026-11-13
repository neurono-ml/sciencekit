//! The ONNX protobuf messages, declared with `prost` derives.
//!
//! Only the fields the exporter writes are declared; the wire tags match the
//! `onnx.proto` / `onnx-ml.proto` definitions, so a runtime decodes the bytes as
//! a regular ONNX model. Declaring the messages here (instead of running
//! `prost-build`) keeps the build free of `protoc` while still producing real
//! protobuf bytes.

/// A serialized ONNX `ModelProto`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(string, tag = "4")]
    pub domain: String,
    #[prost(int64, tag = "5")]
    pub model_version: i64,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<SkOnnxGraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<SkOnnxOperatorSetIdProto>,
    #[prost(message, repeated, tag = "14")]
    pub metadata_props: Vec<SkOnnxStringStringEntryProto>,
}

/// A serialized ONNX `GraphProto`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxGraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<SkOnnxNodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<SkOnnxTensorProto>,
    #[prost(string, tag = "10")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<SkOnnxValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<SkOnnxValueInfoProto>,
}

/// A serialized ONNX `NodeProto`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxNodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<SkOnnxAttributeProto>,
    #[prost(string, tag = "7")]
    pub domain: String,
}

/// A serialized ONNX `AttributeProto`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxAttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes = "vec", tag = "4")]
    pub s: Vec<u8>,
    #[prost(float, repeated, packed = "true", tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, packed = "true", tag = "8")]
    pub ints: Vec<i64>,
    #[prost(bytes = "vec", repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
}

/// A serialized ONNX `TensorProto`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxTensorProto {
    #[prost(int64, repeated, packed = "true", tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
    #[prost(string, tag = "12")]
    pub doc_string: String,
}

/// A serialized ONNX `ValueInfoProto`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<SkOnnxTypeProto>,
    #[prost(string, tag = "3")]
    pub doc_string: String,
}

/// The subset of ONNX `TypeProto` the exporter writes.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxTypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<SkOnnxTensorTypeProto>,
}

/// A serialized ONNX `TypeProto.Tensor`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxTensorTypeProto {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<SkOnnxTensorShapeProto>,
}

/// A serialized ONNX `TensorShapeProto`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxTensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<SkOnnxDimensionProto>,
}

/// A serialized ONNX `TensorShapeProto.Dimension`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxDimensionProto {
    #[prost(int64, optional, tag = "1")]
    pub dim_value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    pub dim_param: Option<String>,
}

/// A serialized ONNX `OperatorSetIdProto`.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxOperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

/// A serialized ONNX `StringStringEntryProto`, used for model metadata.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkOnnxStringStringEntryProto {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, tag = "2")]
    pub value: String,
}
