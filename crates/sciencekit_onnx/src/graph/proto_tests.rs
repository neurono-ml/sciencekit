//! Tests for the protobuf message declarations.

use super::proto;

#[test]
fn proto_node_encodes_domain_and_typed_attributes() {
    let node = proto::SkOnnxNodeProto {
        op_type: "TreeEnsembleClassifier".to_string(),
        domain: "ai.onnx.ml".to_string(),
        input: vec!["features".to_string()],
        output: vec!["label".to_string(), "probabilities".to_string()],
        attribute: vec![proto::SkOnnxAttributeProto {
            name: "post_transform".to_string(),
            s: b"NONE".to_vec(),
            r#type: 3,
            ..Default::default()
        }],
        ..Default::default()
    };
    let bytes = prost::Message::encode_to_vec(&node);
    let decoded = <proto::SkOnnxNodeProto as prost::Message>::decode(bytes.as_slice())
        .expect("a node round-trips");
    assert_eq!(decoded, node);
}

#[test]
fn proto_tensor_uses_raw_little_endian_payload() {
    let tensor = proto::SkOnnxTensorProto {
        dims: vec![2],
        data_type: 11,
        name: "weights".to_string(),
        raw_data: 1.5_f64.to_le_bytes().to_vec(),
        ..Default::default()
    };
    let decoded = <proto::SkOnnxTensorProto as prost::Message>::decode(
        prost::Message::encode_to_vec(&tensor).as_slice(),
    )
    .expect("a tensor round-trips");
    assert_eq!(decoded.data_type, 11);
    assert_eq!(decoded.raw_data, 1.5_f64.to_le_bytes().to_vec());
}
