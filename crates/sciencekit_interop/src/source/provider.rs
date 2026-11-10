//! Adapter from a tabular source to the library's streaming providers (spec
//! `dataset-interop-core`).

use std::marker::PhantomData;

use sciencekit_common::{SKError, SKFloat, SKSourceProvider, SKSupervisedSourceProvider};

use super::lazy_source::{SKSupervisedTabularLazySource, SKTabularLazySource};
use super::SKTabularSource;
use crate::batch::SKFeatureLayout;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;

/// Exposes a [`SKTabularSource`] as the library's streaming providers.
///
/// The provider carries the selection, the missing-value policy and the
/// requested layout, and opens one pass at a time. A reopenable source is
/// reopened on every pass; a one-shot source accepts exactly one open.
pub struct SKTabularSourceProvider<Source, F> {
    source: Source,
    selection: SKColumnSelection,
    policy: SKMissingValuePolicy,
    layout: SKFeatureLayout,
    opened: bool,
    marker: PhantomData<F>,
}

impl<Source, F> SKTabularSourceProvider<Source, F>
where
    Source: SKTabularSource,
{
    /// Wrap a source with the conversion configuration.
    pub fn new(
        source: Source,
        selection: SKColumnSelection,
        policy: SKMissingValuePolicy,
        layout: SKFeatureLayout,
    ) -> Self {
        SKTabularSourceProvider {
            source,
            selection,
            policy,
            layout,
            opened: false,
            marker: PhantomData,
        }
    }

    fn begin_pass(&mut self) -> Result<(), SKError> {
        if !Source::IS_REOPENABLE && self.opened {
            return Err(SKError::UnsupportedOperation {
                operation: "open_tabular_source",
                reason: "the source is one-shot and was already opened".to_string(),
            });
        }
        self.source.open()?;
        self.opened = true;
        Ok(())
    }
}

impl<F, Source> SKSourceProvider<F> for SKTabularSourceProvider<Source, F>
where
    F: SKFloat,
    Source: SKTabularSource,
{
    type Error = SKError;

    type Source<'pass>
        = SKTabularLazySource<'pass, Source, F>
    where
        Self: 'pass;

    const IS_REOPENABLE: bool = Source::IS_REOPENABLE;

    fn open(&mut self) -> Result<Self::Source<'_>, SKError> {
        self.begin_pass()?;
        Ok(SKTabularLazySource {
            source: &mut self.source,
            selection: &self.selection,
            policy: self.policy,
            layout: self.layout,
            marker: PhantomData,
        })
    }
}

impl<F, Source> SKSupervisedSourceProvider<F> for SKTabularSourceProvider<Source, F>
where
    F: SKFloat,
    Source: SKTabularSource,
{
    type Error = SKError;

    type Source<'pass>
        = SKSupervisedTabularLazySource<'pass, Source, F>
    where
        Self: 'pass;

    const IS_REOPENABLE: bool = Source::IS_REOPENABLE;

    fn open(&mut self) -> Result<Self::Source<'_>, SKError> {
        if !self.selection.has_target() {
            return Err(SKError::UnsupportedOperation {
                operation: "supervised_tabular_source",
                reason: "the selection does not declare a target column".to_string(),
            });
        }
        self.begin_pass()?;
        Ok(SKSupervisedTabularLazySource {
            source: &mut self.source,
            selection: &self.selection,
            policy: self.policy,
            layout: self.layout,
            marker: PhantomData,
        })
    }
}
