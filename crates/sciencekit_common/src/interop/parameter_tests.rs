//! Tests for typed hyperparameter values (spec `model-export-contracts`).

use super::SKParameterValue;

/// Every scalar variant reports its own typed accessor and nothing else.
#[test]
fn interop_state_parameter_variants_are_typed() {
    let boolean = SKParameterValue::Bool(true);
    let integer = SKParameterValue::Integer(7);
    let float = SKParameterValue::Float(0.25);
    let text = SKParameterValue::String("l2".to_string());
    let list = SKParameterValue::List(vec![
        SKParameterValue::Integer(1),
        SKParameterValue::Integer(2),
    ]);

    assert_eq!(boolean.as_bool(), Some(true));
    assert_eq!(integer.as_integer(), Some(7));
    assert_eq!(float.as_float(), Some(0.25));
    assert_eq!(text.as_string(), Some("l2"));
    assert_eq!(list.as_list().map(<[SKParameterValue]>::len), Some(2));

    assert_eq!(boolean.as_integer(), None);
    assert_eq!(integer.as_float(), None);
    assert_eq!(float.as_string(), None);
    assert_eq!(text.as_list(), None);
    assert_eq!(list.as_bool(), None);
}

/// Conversions build the matching variant without naming it.
#[test]
fn interop_state_parameter_conversions_match_the_value() {
    assert_eq!(SKParameterValue::from(true), SKParameterValue::Bool(true));
    assert_eq!(SKParameterValue::from(3_i64), SKParameterValue::Integer(3));
    assert_eq!(
        SKParameterValue::from(1.5_f64),
        SKParameterValue::Float(1.5)
    );
    assert_eq!(
        SKParameterValue::from("mock"),
        SKParameterValue::String("mock".to_string())
    );
    assert_eq!(
        SKParameterValue::from("mock".to_string()),
        SKParameterValue::String("mock".to_string())
    );
}
