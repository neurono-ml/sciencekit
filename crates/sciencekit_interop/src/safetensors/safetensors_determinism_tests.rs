//! Deterministic single-file encoding tests (spec `safetensors-format`).
//!
//! A model artifact is an experiment record: writing the same fitted state
//! twice must produce the same bytes, so two runs of one workflow can be
//! compared with a plain file diff.

use sciencekit_common::{SKModelState, SKParameterValue, SKTensor};

use super::single_file::encode_single_file_bytes;

fn sk_state() -> SKModelState<f64> {
    let metadata = sciencekit_common::SKModelMetadata::new(
        "SKRidge",
        1,
        sciencekit_common::SKTensorDataType::F64,
        2,
    )
    .expect("the metadata builds")
    .with_parameter("backend", SKParameterValue::from("faer"))
    .expect("the backend parameter")
    .with_parameter("alpha", SKParameterValue::from(1.0))
    .expect("the alpha parameter")
    .with_parameter("include_intercept", SKParameterValue::from(true))
    .expect("the intercept parameter");
    let coefficients = SKTensor::floating("ridge.coefficients", vec![1, 2], vec![1.0, 2.0])
        .expect("the coefficients");
    let intercepts =
        SKTensor::floating("ridge.intercepts", vec![1], vec![0.5]).expect("the intercepts");
    SKModelState::new(metadata, vec![coefficients, intercepts]).expect("the state")
}

fn sk_header(bytes: &[u8]) -> &str {
    let header_length = u64::from_le_bytes(bytes[..8].try_into().expect("eight length bytes"));
    let header = &bytes[8..8 + header_length as usize];
    std::str::from_utf8(header).expect("a UTF-8 header")
}

#[test]
fn the_single_file_encoding_is_identical_for_the_same_state() {
    let state = sk_state();
    let mut encoded = Vec::new();
    for _ in 0..32 {
        encoded.push(encode_single_file_bytes(&state).expect("the state encodes"));
    }
    for other in &encoded[1..] {
        assert_eq!(
            encoded[0], *other,
            "every encoding of one state must be byte-identical"
        );
    }
}

#[test]
fn the_header_orders_metadata_alphabetically_and_tensors_by_name() {
    let state = sk_state();
    let bytes = encode_single_file_bytes(&state).expect("the state encodes");
    let header = sk_header(&bytes);
    assert!(
        header.starts_with(r#"{"__metadata__":{"sciencekit.estimator_format_version":"#),
        "metadata keys must be sorted, header was {header}"
    );
    let coefficients = header.find("ridge.coefficients").expect("the coefficients");
    let intercepts = header.find("ridge.intercepts").expect("the intercepts");
    assert!(
        coefficients < intercepts,
        "tensor descriptors must follow tensor-name order, header was {header}"
    );
}
