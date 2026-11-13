//! Tree-ensemble prediction over host views (spec `opencl-backend`).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKError, SKFloat};

use super::super::implementation::SKOpenClBackend;
use super::support::{element_bytes, ensure_scalar_supported};
use crate::opencl::buffer;
use crate::opencl::kernels::{self, SK_OPENCL_TREE_NODE_COLUMNS};
use crate::opencl::observability;

impl SKOpenClBackend {
    /// Tree-ensemble prediction.
    ///
    /// `nodes` carries one tree per `node_count` rows, each row with
    /// [`SK_OPENCL_TREE_NODE_COLUMNS`] columns: feature index, threshold, left
    /// child, right child, leaf value and leaf flag (`1` for a leaf).
    pub fn predict_trees<F: SKFloat>(
        &self,
        features: ArrayView2<'_, F>,
        nodes: ArrayView2<'_, F>,
    ) -> Result<Array2<F>, SKError> {
        let span = observability::operation_span("tree_predict");
        let _entered = span.enter();
        let session = self.session()?;
        ensure_scalar_supported::<F>(session)?;
        if nodes.ncols() != SK_OPENCL_TREE_NODE_COLUMNS {
            return Err(SKError::ShapeMismatch {
                expected: vec![nodes.nrows(), SK_OPENCL_TREE_NODE_COLUMNS],
                found: vec![nodes.nrows(), nodes.ncols()],
            });
        }
        let feature_resident = buffer::upload_view(session, features)?;
        let node_resident = buffer::upload_view(session, nodes)?;
        let output = buffer::allocate_resident::<F>(session, (features.nrows(), 1))?;
        kernels::run_tree_predict(session, &feature_resident, &node_resident, &output)?;
        let result = buffer::download_buffer(session, &output)?;
        let upload_bytes = element_bytes::<F>(features.len() + nodes.len());
        let output_bytes = element_bytes::<F>(output.element_count());
        self.charge(upload_bytes, output_bytes);
        observability::record_device(&span, session.device().index);
        observability::record_transfer(&span, upload_bytes + output_bytes);
        Ok(result)
    }

    /// Account one operation's transfers.
    pub(crate) fn charge(&self, upload_bytes: u64, kernel_bytes: u64) {
        use std::sync::atomic::Ordering;
        self.upload_bytes.fetch_add(upload_bytes, Ordering::Relaxed);
        self.kernel_bytes.fetch_add(kernel_bytes, Ordering::Relaxed);
        self.download_bytes
            .fetch_add(kernel_bytes, Ordering::Relaxed);
    }
}
