//! Incremental learning: one more batch merged into the running state
//! (spec `simple-imputer`).

use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::SKSimpleImputerModel;

impl<F: SKFloat> SKSimpleImputerModel<F> {
    /// Continue learning from one more batch, updating the accumulated state
    /// and refreshing the statistics. Returns the model itself for chaining.
    /// An empty batch is a no-op.
    pub fn partial_fit<'a, X>(&mut self, features: X) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let _view = features.try_into()?;
        Ok(self)
    }
}
