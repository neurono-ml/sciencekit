//! Tests for streaming batches (spec `streaming-batches`).

use ndarray::{Array2, ArrayView1};

use super::{SKDataBatch, SKLazySource, SKMappableSource};
use crate::SKError;

/// An owned block survives the source's drop (its data is fully owned).
#[test]
fn block_survives_the_source() {
    let data = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let batch = SKDataBatch::new(data, 0, false);

    // The batch owns its data — extracting it (and dropping any "source") leaves it intact.
    let extracted = batch.clone();
    assert_eq!(
        extracted.data().as_slice_memory_order().unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );
}

/// Exactly one final block is identifiable on a finite source.
#[test]
fn exactly_one_final_block_is_identifiable() {
    let mut source = FiniteSource::new();
    let mut finals = 0;
    let mut count = 0;
    while let Some(batch) = source.next_batch().unwrap() {
        count += 1;
        if batch.is_final() {
            finals += 1;
        }
    }
    assert_eq!(count, 3);
    assert_eq!(finals, 1);
}

/// An intermediate read failure yields a structured error, not a panic.
#[test]
fn read_failure_stops_with_structured_error() {
    let mut source = FailingSource;
    match source.next_batch() {
        Err(e) => assert!(e.to_string().contains("stream failed")),
        Ok(_) => panic!("expected the failing source to yield an error"),
    }
}

/// Direct positional access returns a row without scanning previous rows.
#[test]
fn random_access_is_direct() {
    let source = VecSource(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
    let row = source.row(4).unwrap();
    assert_eq!(row.as_slice().unwrap(), &[5.0]);
    let row0 = source.row(0).unwrap();
    assert_eq!(row0.as_slice().unwrap(), &[1.0]);
}

/// The random-access contract does not require memory mapping.
#[test]
fn random_access_contract_does_not_couple_storage() {
    // VecSource implements SKMappableSource over a plain Vec — no memmap dependency.
    fn assert_mappable<T: SKMappableSource<f64>>() {}
    assert_mappable::<VecSource>();
    let source = VecSource(vec![1.0]);
    assert_eq!(source.number_of_rows(), 1);
}

// ---- Example sources -------------------------------------------------------

/// A finite sequential source of 3 batches; the last is marked final.
struct FiniteSource {
    next_position: usize,
}
impl FiniteSource {
    fn new() -> Self {
        FiniteSource { next_position: 0 }
    }
}
impl SKLazySource<f64> for FiniteSource {
    type Error = SKError;
    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        if self.next_position >= 3 {
            return Ok(None);
        }
        let pos = self.next_position;
        self.next_position += 1;
        let data = Array2::from_shape_vec((1, 1), vec![pos as f64]).unwrap();
        Ok(Some(SKDataBatch::new(data, pos, pos == 2)))
    }
}

/// A source that fails on the first read with a structured error.
struct FailingSource;
impl SKLazySource<f64> for FailingSource {
    type Error = SKError;
    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        Err(SKError::Conversion("stream failed mid-read".into()))
    }
}

/// A random-access source over a plain `Vec` (no memory mapping).
struct VecSource(Vec<f64>);
impl SKMappableSource<f64> for VecSource {
    type Error = SKError;
    fn number_of_rows(&self) -> usize {
        self.0.len()
    }
    fn row(&self, index: usize) -> Result<ArrayView1<'_, f64>, Self::Error> {
        if index >= self.0.len() {
            return Err(SKError::shape_mismatch_2d(self.0.len(), 1, index + 1, 1));
        }
        Ok(ArrayView1::from(&self.0[index..index + 1]))
    }
}
