//! The feature-only source provider contract (spec
//! `streaming-source-provider`).

use crate::SKError;

use super::super::SKLazySource;

/// A supplier that opens one feature-only streaming pass at a time.
///
/// A concrete sequential source satisfies this contract through the residual
/// implementation for `&mut S`, reporting [`Self::IS_REOPENABLE`] as `false`;
/// [`SKReopenableSource`](super::SKReopenableSource) wraps a factory and opens a
/// fresh source per pass. The generic associated type lets both shapes share one
/// trait without dynamic dispatch: the one-shot implementation returns a
/// reborrow of the source, the reopenable implementation an owned source.
pub trait SKSourceProvider<F> {
    /// The source opened for one pass.
    type Source<'pass>: SKLazySource<F> + Send
    where
        Self: 'pass;

    /// Whether the provider can open more than one pass.
    const IS_REOPENABLE: bool;

    /// Open one pass, returning a source that yields blocks until its end.
    ///
    /// Opening is fallible and reports failures through the central taxonomy.
    fn open(&mut self) -> Result<Self::Source<'_>, SKError>;
}

impl<F, Source> SKSourceProvider<F> for &mut Source
where
    Source: SKLazySource<F> + Send,
{
    type Source<'pass>
        = &'pass mut Source
    where
        Self: 'pass;

    const IS_REOPENABLE: bool = false;

    fn open(&mut self) -> Result<Self::Source<'_>, SKError> {
        Ok(&mut **self)
    }
}
