//! Companion tests for the shared export conventions.

use crate::SKBackendKind;
use crate::interop::shared::{
    SK_MODEL_FORMAT_VERSION, sk_backend_parameter, sk_boolean_parameter, sk_float_parameter,
    sk_integer_parameter, sk_model_tensor_data_type, sk_parse_backend_name,
    sk_parse_backend_parameter, sk_string_parameter,
};
use crate::interop::{SKModelMetadata, SKTensorDataType};

#[test]
fn the_format_version_is_one() {
    assert_eq!(SK_MODEL_FORMAT_VERSION, 1);
}

#[test]
fn scalar_mapping_matches_the_scalar_width() {
    assert_eq!(sk_model_tensor_data_type::<f64>(), SKTensorDataType::F64);
    assert_eq!(sk_model_tensor_data_type::<f32>(), SKTensorDataType::F32);
}

#[test]
fn backend_parameter_round_trips_every_kind() {
    for kind in SKBackendKind::ALL {
        let encoded = sk_backend_parameter(kind);
        let name = encoded.as_string().expect("a backend encodes as a string");
        assert_eq!(sk_parse_backend_name(name), kind);
        assert_eq!(sk_parse_backend_name(&format!("{kind:?}")), kind);
    }
}

#[test]
fn metadata_readers_default_when_absent() {
    let metadata =
        SKModelMetadata::new("SKTest", SK_MODEL_FORMAT_VERSION, SKTensorDataType::F64, 2).unwrap();
    assert_eq!(sk_integer_parameter(&metadata, "missing", 7), 7);
    assert!(sk_boolean_parameter(&metadata, "missing", true));
    assert_eq!(sk_float_parameter(&metadata, "missing", 0.5), 0.5);
    assert_eq!(
        sk_string_parameter(&metadata, "missing", "solver"),
        "solver"
    );
    assert_eq!(sk_parse_backend_parameter(&metadata), SKBackendKind::Faer);
}
