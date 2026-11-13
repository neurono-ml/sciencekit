//! A registry of adapters addressed by identifier (spec `lora-adapters`,
//! requirement "Validation and concurrency").

use std::collections::{BTreeMap, btree_map};

use sciencekit_common::{SKError, SKFloat};

use super::attached_model::SKLoraAttachedModel;

/// Adapters keyed by their stable identifier.
///
/// The registry rejects a duplicate identifier at insertion time, so a
/// collection of adapters never silently replaces one task's adapter with
/// another's. It is `Send + Sync` by construction (the map and the attached
/// models are), so several threads can read the same registry.
#[derive(Debug, Clone, Default)]
pub struct SKLoraAdapterRegistry<F> {
    entries: BTreeMap<String, SKLoraAttachedModel<F>>,
}

impl<F: SKFloat> SKLoraAdapterRegistry<F> {
    /// An empty registry.
    pub fn new() -> SKLoraAdapterRegistry<F> {
        SKLoraAdapterRegistry {
            entries: BTreeMap::new(),
        }
    }

    /// Insert an adapter under its identifier.
    ///
    /// # Errors
    ///
    /// Returns [`SKError::InvalidHyperparameter`] when the identifier is
    /// already taken.
    pub fn insert(&mut self, adapter: SKLoraAttachedModel<F>) -> Result<(), SKError> {
        let identifier = adapter.adapter().identifier().to_string();
        match self.entries.entry(identifier.clone()) {
            btree_map::Entry::Occupied(_) => Err(SKError::InvalidHyperparameter {
                name: "identifier",
                reason: format!("adapter identifier `{identifier}` is not unique"),
            }),
            btree_map::Entry::Vacant(slot) => {
                slot.insert(adapter);
                Ok(())
            }
        }
    }

    /// The adapter stored under an identifier.
    pub fn get(&self, identifier: &str) -> Option<&SKLoraAttachedModel<F>> {
        self.entries.get(identifier)
    }

    /// Every identifier, in deterministic order.
    pub fn identifiers(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// The number of stored adapters.
    pub fn number_of_adapters(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
