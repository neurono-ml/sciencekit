//! TDD suite for the Arrow schema mapping (task 1.2).

use arrow_schema::{DataType, Field, Schema, TimeUnit};

use super::schema_mapping::{
    SKArrowColumnRoles, sk_arrow_schema_from_tabular_schema, sk_tabular_schema_from_arrow_schema,
};
use crate::schema::{SKColumnRole, SKLogicalType, SKTabularSchema};

fn roles() -> SKArrowColumnRoles {
    SKArrowColumnRoles::new()
}

#[test]
fn sk_primitive_and_text_types_map_to_neutral_logical_types() {
    let schema = Schema::new(vec![
        Field::new("single", DataType::Float32, true),
        Field::new("double", DataType::Float64, true),
        Field::new("count", DataType::Int32, true),
        Field::new("big_count", DataType::Int64, true),
        Field::new("flag", DataType::Boolean, true),
        Field::new("species", DataType::Utf8, true),
    ]);

    let tabular =
        sk_tabular_schema_from_arrow_schema(&schema, &roles()).expect("a supported schema");

    let types: Vec<SKLogicalType> = tabular
        .columns()
        .iter()
        .map(|column| column.logical_type())
        .collect();
    assert_eq!(
        types,
        vec![
            SKLogicalType::Float32,
            SKLogicalType::Float64,
            SKLogicalType::Integer64,
            SKLogicalType::Integer64,
            SKLogicalType::Boolean,
            SKLogicalType::Nominal,
        ]
    );
    assert_eq!(
        tabular.column("species").expect("the column").name(),
        "species"
    );
}

#[test]
fn sk_string_dictionary_maps_to_category_codes() {
    let schema = Schema::new(vec![Field::new(
        "class",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        true,
    )]);

    let tabular = sk_tabular_schema_from_arrow_schema(&schema, &roles()).expect("a dictionary");

    assert_eq!(
        tabular.column("class").expect("the column").logical_type(),
        SKLogicalType::CategoryCode
    );
}

#[test]
fn sk_nested_temporal_and_binary_columns_are_rejected_with_the_field_name() {
    let unsupported = [
        (
            DataType::List(Field::new("item", DataType::Float64, true).into()),
            "nested",
        ),
        (DataType::Timestamp(TimeUnit::Millisecond, None), "temporal"),
        (DataType::Binary, "binary"),
        (DataType::Decimal128(10, 2), "decimal"),
    ];

    for (data_type, expected_hint) in unsupported {
        let schema = Schema::new(vec![Field::new("problem", data_type, true)]);
        let error = sk_tabular_schema_from_arrow_schema(&schema, &roles())
            .expect_err("an unsupported type must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("problem"),
            "the error must name the field: {message}"
        );
        assert!(
            message.contains(expected_hint),
            "the error must describe the rejected family (`{expected_hint}`): {message}"
        );
        assert!(
            message.contains("convert") || message.contains("cast") || message.contains("flatten"),
            "the error must suggest a conversion: {message}"
        );
    }
}

#[test]
fn sk_roles_come_from_the_request_and_from_field_metadata() {
    let schema = Schema::new(vec![
        Field::new("feature", DataType::Float64, true),
        Field::new("engine_target", DataType::Float64, true),
        Field::new("annotated", DataType::Float64, true).with_metadata(
            std::collections::HashMap::from([(
                super::schema_mapping::SK_ARROW_ROLE_METADATA_KEY.to_string(),
                "ignored".to_string(),
            )]),
        ),
    ]);

    // `annotated` carries BOTH a `sciencekit:role` field metadata entry
    // (`ignored`) and a conflicting request entry (`target`); field metadata
    // takes precedence over the request.
    let requested_roles = roles()
        .target("engine_target")
        .role("annotated", SKColumnRole::Target);
    let tabular =
        sk_tabular_schema_from_arrow_schema(&schema, &requested_roles).expect("roles resolve");

    assert_eq!(
        tabular.column("feature").expect("column").role(),
        SKColumnRole::Feature
    );
    assert_eq!(
        tabular.column("engine_target").expect("column").role(),
        SKColumnRole::Target
    );
    assert_eq!(
        tabular.column("annotated").expect("column").role(),
        SKColumnRole::Ignored
    );
}

#[test]
fn sk_round_trip_preserves_names_types_and_roles() {
    let original = SKTabularSchema::new(vec![
        crate::schema::SKColumnSchema::new("a", SKLogicalType::Float32, SKColumnRole::Feature),
        crate::schema::SKColumnSchema::new("b", SKLogicalType::Integer64, SKColumnRole::Target),
        crate::schema::SKColumnSchema::new("c", SKLogicalType::Nominal, SKColumnRole::Ignored),
        crate::schema::SKColumnSchema::new("d", SKLogicalType::CategoryCode, SKColumnRole::Feature),
    ])
    .expect("a valid schema");

    let arrow_schema = sk_arrow_schema_from_tabular_schema(&original).expect("exportable");

    assert_eq!(arrow_schema.fields().len(), 4);
    let restored =
        sk_tabular_schema_from_arrow_schema(&arrow_schema, &roles()).expect("importable");
    assert_eq!(restored, original);
}

#[test]
fn sk_record_batch_import_with_roles_applies_caller_role_assignments() {
    use std::sync::Arc;
    let schema = Arc::new(Schema::new(vec![
        Field::new("single", DataType::Float32, false),
        Field::new("count", DataType::Int64, false),
        Field::new("species", DataType::Utf8, false),
    ]));
    let batch = arrow_array::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Float32Array::from(vec![1.5_f32])),
            Arc::new(arrow_array::Int64Array::from(vec![2_i64])),
            Arc::new(arrow_array::StringArray::from(vec!["a"])),
        ],
    )
    .expect("a batch");
    let assigned = roles()
        .target("count")
        .role("species", SKColumnRole::Ignored);
    let tabular = super::import::sk_tabular_batch_from_record_batch_with_roles(
        &batch,
        &assigned,
        crate::missing::SKMissingValuePolicy::Reject,
        0,
        true,
    )
    .expect("import with explicit roles");
    assert_eq!(
        tabular.schema().column("single").unwrap().role(),
        SKColumnRole::Feature
    );
    assert_eq!(
        tabular.schema().column("count").unwrap().role(),
        SKColumnRole::Target
    );
    assert_eq!(
        tabular.schema().column("species").unwrap().role(),
        SKColumnRole::Ignored
    );
}
