//! Tests for label canonicalization (spec `data-view-boundary`).

use super::{SKLabelTable, sk_canonicalize_labels};

/// Roundtrip: canonicalize then decode via the table restores the original.
#[test]
fn roundtrip_preserves_original_labels() {
    let labels = ["cat", "dog", "cat", "bird", "dog"];
    let (indices, table) = sk_canonicalize_labels(&labels);

    // Decode indices back through the table.
    let restored: Vec<&str> = indices
        .iter()
        .map(|&i| table.label_of(i).unwrap())
        .collect();
    assert_eq!(restored, labels);
}

/// Same input canonicalized twice yields identical mappings.
#[test]
fn same_input_produces_same_table() {
    let labels = ["a", "b", "a", "c"];
    let (indices_a, table_a) = sk_canonicalize_labels(&labels);
    let (indices_b, table_b) = sk_canonicalize_labels(&labels);

    assert_eq!(indices_a, indices_b);
    assert_eq!(table_a, table_b);
}

/// The table exposes readable metadata (label↔index access, class count).
#[test]
fn table_is_bearer_of_exportable_metadata() {
    let labels = ["x", "y", "x"];
    let (indices, table) = sk_canonicalize_labels(&labels);

    assert_eq!(indices, vec![0, 1, 0]);
    assert_eq!(table.number_of_classes(), 2);
    assert_eq!(table.index_of("x"), Some(0));
    assert_eq!(table.index_of("y"), Some(1));
    assert_eq!(table.label_of(1), Some("y"));
    assert_eq!(table.index_of("missing"), None);
    let collected: Vec<&str> = table.labels().collect();
    assert_eq!(collected, vec!["x", "y"]);
}

/// Canonicalization is compact: first-occurrence order defines the indices.
#[test]
fn indices_are_compact_first_occurrence() {
    let labels = ["z", "a", "z", "b", "a"];
    let (indices, table) = sk_canonicalize_labels(&labels);
    assert_eq!(indices, vec![0, 1, 0, 2, 1]);
    assert_eq!(table.number_of_classes(), 3);
}

/// `SKLabelTable` is `Send + Sync` for use across threads (model metadata).
#[test]
fn table_is_thread_shareable() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SKLabelTable>();
}
