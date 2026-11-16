//! Tests for the access-pattern declaration contract (spec `access-pattern`).

use super::{SKAccessPattern, SKDeclaredAccessPattern};

struct SequentialStub;

impl SKDeclaredAccessPattern for SequentialStub {
    fn access_pattern(&self) -> SKAccessPattern {
        SKAccessPattern::Sequential
    }
}

struct FullMemoryStub;

impl SKDeclaredAccessPattern for FullMemoryStub {
    fn access_pattern(&self) -> SKAccessPattern {
        SKAccessPattern::FullMemory
    }

    fn approximate_alternative(&self) -> Option<&'static str> {
        Some("RandomizedSVD")
    }
}

#[test]
fn a_declaration_is_readable_without_data() {
    assert_eq!(SequentialStub.access_pattern(), SKAccessPattern::Sequential);
    assert_eq!(FullMemoryStub.access_pattern(), SKAccessPattern::FullMemory);
}

#[test]
fn the_default_alternative_is_none() {
    assert_eq!(SequentialStub.approximate_alternative(), None);
}

#[test]
fn a_full_memory_declaration_names_its_alternative() {
    assert_eq!(
        FullMemoryStub.approximate_alternative(),
        Some("RandomizedSVD")
    );
}
