//! Tests for the owned streaming feature block (spec
//! `sparse-streaming-batches`).

use ndarray::Array2;
use sprs::CsMat;

use super::SKFeatureBlock;
use crate::SKDataView;

fn dense_block() -> SKFeatureBlock<f64> {
    let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    SKFeatureBlock::Dense(matrix)
}

/// A sparse block with two non-zeros in a large shape: the dense figure would
/// be `rows × columns × size_of`, far above the stored CSR arrays.
fn sparse_block() -> SKFeatureBlock<f64> {
    // Two non-zeros in a 200 × 400 shape: the dense figure dwarfs the CSR arrays.
    let mut indptr = vec![1_usize; 201];
    indptr[0] = 0;
    indptr[200] = 2;
    let matrix = CsMat::new((200, 400), indptr, vec![1, 399], vec![10.0_f64, 20.0]);
    SKFeatureBlock::Sparse(matrix)
}

#[test]
fn dense_block_reports_its_shape() {
    let block = dense_block();
    assert_eq!(block.number_of_rows(), 2);
    assert_eq!(block.number_of_columns(), 3);
    assert!(!block.is_sparse());
}

#[test]
fn sparse_block_reports_its_shape() {
    let block = sparse_block();
    assert_eq!(block.number_of_rows(), 200);
    assert_eq!(block.number_of_columns(), 400);
    assert!(block.is_sparse());
}

/// The dense view borrows the block's own buffer — same address, no copy.
#[test]
fn dense_view_is_zero_copy() {
    let block = dense_block();
    let stored = match &block {
        SKFeatureBlock::Dense(matrix) => matrix.as_ptr(),
        SKFeatureBlock::Sparse(_) => panic!("expected dense"),
    };
    match block.view() {
        SKDataView::Dense(view) => assert_eq!(view.as_ptr(), stored),
        SKDataView::Sparse(_) => panic!("expected a dense view"),
    }
}

/// The sparse view borrows the block's CSR arrays and allocates no dense
/// matrix.
#[test]
fn sparse_view_is_zero_copy() {
    let block = sparse_block();
    let stored = match &block {
        SKFeatureBlock::Sparse(matrix) => matrix.data().as_ptr(),
        SKFeatureBlock::Dense(_) => panic!("expected sparse"),
    };
    match block.view() {
        SKDataView::Sparse(view) => {
            assert_eq!(view.data().as_ptr(), stored);
            assert_eq!(view.nnz(), 2);
        }
        SKDataView::Dense(_) => panic!("expected a sparse view"),
    }
}

#[test]
fn dense_byte_size_is_rows_times_columns_times_scalar() {
    let block = dense_block();
    assert_eq!(block.byte_size(), 2 * 3 * size_of::<f64>());
}

/// The sparse size counts the element, index and pointer arrays, not the dense
/// `rows × columns × size_of` figure.
#[test]
fn sparse_byte_size_follows_the_stored_buffers() {
    let block = sparse_block();
    let expected = 2 * size_of::<f64>() + 2 * size_of::<usize>() + 201 * size_of::<usize>();
    assert_eq!(block.byte_size(), expected);

    let dense_figure = block.number_of_rows() * block.number_of_columns() * size_of::<f64>();
    assert!(
        block.byte_size() < dense_figure / 2,
        "sparse size {} should be far below the dense figure {}",
        block.byte_size(),
        dense_figure
    );
}

/// A block fully owns its data: dropping the source leaves it intact.
#[test]
fn block_owns_its_data() {
    let block = {
        let matrix = CsMat::new((2, 2), vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]);
        SKFeatureBlock::Sparse(matrix)
    };
    assert_eq!(block.number_of_rows(), 2);
    assert_eq!(
        block.byte_size(),
        2 * size_of::<f64>() + 2 * size_of::<usize>() + 3 * size_of::<usize>()
    );
}

/// A block and its representation survive a move to another thread.
#[test]
fn block_moves_across_threads() {
    let block = sparse_block();
    let handle = std::thread::spawn(move || {
        assert!(block.is_sparse());
        assert_eq!(block.number_of_rows(), 200);
        match block.view() {
            SKDataView::Sparse(view) => view.nnz(),
            SKDataView::Dense(_) => panic!("the representation must survive the move"),
        }
    });
    assert_eq!(handle.join().unwrap(), 2);
}

/// f32 blocks report the same shape and representation as f64 blocks.
#[test]
fn block_supports_f32() {
    let dense = SKFeatureBlock::Dense(Array2::from_shape_vec((2, 3), vec![1.0_f32; 6]).unwrap());
    assert_eq!(dense.byte_size(), 6 * size_of::<f32>());
    assert!(!dense.is_sparse());

    let sparse = SKFeatureBlock::Sparse(CsMat::new(
        (2, 3),
        vec![0, 1, 2],
        vec![1, 2],
        vec![1.0_f32, 2.0],
    ));
    assert_eq!(sparse.number_of_rows(), 2);
    assert_eq!(sparse.number_of_columns(), 3);
    assert_eq!(
        sparse.byte_size(),
        2 * size_of::<f32>() + 2 * size_of::<usize>() + 3 * size_of::<usize>()
    );
}
