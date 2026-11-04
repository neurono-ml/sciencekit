//! The reopenable source provider wrapping a fallible factory (spec
//! `streaming-source-provider`).

use crate::SKError;

use super::super::{SKLazySource, SKSupervisedLazySource};
use super::{SKSourceProvider, SKSupervisedSourceProvider};

/// A provider that opens a fresh source per pass from a fallible factory.
///
/// The wrapped `FnMut() -> Result<S, SKError>` is invoked once per
/// [`open`](SKSourceProvider::open) call, so every pass reads the same data in
/// the same order. A factory error aborts the fit with the original cause.
/// Wrapping happens once at construction:
/// `SKReopenableSource::new(|| FileSource::open(path))`.
pub struct SKReopenableSource<Factory> {
    factory: Factory,
}

impl<Factory> SKReopenableSource<Factory> {
    /// Wrap a factory that produces one source per pass.
    pub fn new(factory: Factory) -> Self {
        SKReopenableSource { factory }
    }
}

impl<F, Source, Factory> SKSourceProvider<F> for SKReopenableSource<Factory>
where
    Source: SKLazySource<F> + Send,
    Factory: FnMut() -> Result<Source, SKError>,
{
    type Source<'pass>
        = Source
    where
        Self: 'pass;

    const IS_REOPENABLE: bool = true;

    fn open(&mut self) -> Result<Self::Source<'_>, SKError> {
        (self.factory)()
    }
}

impl<F, Source, Factory> SKSupervisedSourceProvider<F> for SKReopenableSource<Factory>
where
    Source: SKSupervisedLazySource<F> + Send,
    Factory: FnMut() -> Result<Source, SKError>,
{
    type Source<'pass>
        = Source
    where
        Self: 'pass;

    const IS_REOPENABLE: bool = true;

    fn open(&mut self) -> Result<Self::Source<'_>, SKError> {
        (self.factory)()
    }
}
