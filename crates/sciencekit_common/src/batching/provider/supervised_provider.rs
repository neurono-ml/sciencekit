//! The supervised source provider contract (spec
//! `streaming-source-provider`).

use crate::SKError;

use super::super::SKSupervisedLazySource;

/// A supplier that opens one supervised streaming pass at a time.
///
/// The supervised twin of [`SKSourceProvider`](super::SKSourceProvider): each
/// block pairs a feature matrix with its targets. A concrete source satisfies
/// the contract through the residual implementation for `&mut S`, reporting
/// [`Self::IS_REOPENABLE`] as `false`;
/// [`SKReopenableSource`](super::SKReopenableSource) wraps a factory and opens a
/// fresh source per pass.
pub trait SKSupervisedSourceProvider<F> {
    /// The iteration error of every source this provider opens.
    ///
    /// The executor maps it into the central [`SKError`] to report a read
    /// failure, so it must convert into the taxonomy. The source trait itself
    /// keeps only the opposite direction ([`SKSupervisedLazySource::Error`]).
    type Error: Into<SKError>;

    /// The source opened for one pass.
    type Source<'pass>: SKSupervisedLazySource<F, Error = Self::Error> + Send
    where
        Self: 'pass;

    /// Whether the provider can open more than one pass.
    const IS_REOPENABLE: bool;

    /// Open one pass, returning a source that yields blocks until its end.
    ///
    /// Opening is fallible and reports failures through the central taxonomy.
    fn open(&mut self) -> Result<Self::Source<'_>, SKError>;
}

impl<F, Source> SKSupervisedSourceProvider<F> for &mut Source
where
    Source: SKSupervisedLazySource<F> + Send,
    Source::Error: Into<SKError>,
{
    type Error = Source::Error;

    type Source<'pass>
        = &'pass mut Source
    where
        Self: 'pass;

    const IS_REOPENABLE: bool = false;

    fn open(&mut self) -> Result<Self::Source<'_>, SKError> {
        Ok(&mut **self)
    }
}
