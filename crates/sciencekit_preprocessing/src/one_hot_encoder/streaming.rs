//! The out-of-core accumulator for `SKOneHotEncoder::fit_streaming`.

use sciencekit_common::{
    SK_CATEGORICAL_MISSING_CODE, SKCategoricalView, SKError, sk_canonicalize_labels,
};

use crate::one_hot_encoder::categories::{SKIntegerCategoryTable, SKOneHotColumnCategories};

/// Running accumulator for `fit_streaming`: one column of labels (nominal) or
/// one code table (integer) per input column, in batch order.
enum StreamingCategories {
    Nominal(Vec<Vec<String>>),
    Integer(Vec<SKIntegerCategoryTable>),
}

impl StreamingCategories {
    fn finalize(self) -> Result<Vec<SKOneHotColumnCategories>, SKError> {
        match self {
            StreamingCategories::Nominal(columns) => Ok(columns
                .iter()
                .map(|labels| {
                    let borrowed: Vec<&str> = labels.iter().map(String::as_str).collect();
                    let (_indices, table) = sk_canonicalize_labels(&borrowed);
                    SKOneHotColumnCategories::Nominal(table)
                })
                .collect()),
            StreamingCategories::Integer(tables) => Ok(tables
                .into_iter()
                .map(SKOneHotColumnCategories::Integer)
                .collect()),
        }
    }
}

/// Accumulate every batch into running column state, enforcing a consistent
/// column count and representation.
pub(super) fn accumulate_batches<'a, I, C>(
    batches: I,
) -> Result<Vec<SKOneHotColumnCategories>, SKError>
where
    I: IntoIterator<Item = C>,
    C: TryInto<SKCategoricalView<'a>, Error = SKError>,
{
    let mut accumulator: Option<StreamingCategories> = None;
    let mut number_of_columns: Option<usize> = None;

    for batch in batches {
        let view = batch.try_into()?;
        if view.number_of_rows() == 0 {
            continue;
        }
        let (rows_count, columns_count) = view.shape();
        if columns_count == 0 {
            return Err(SKError::shape_mismatch_2d(rows_count, 1, rows_count, 0));
        }
        match number_of_columns {
            None => number_of_columns = Some(columns_count),
            Some(expected) if expected != columns_count => {
                return Err(SKError::ShapeMismatch {
                    expected: vec![expected],
                    found: vec![columns_count],
                });
            }
            _ => {}
        }
        accumulate_one(&view, rows_count, columns_count, &mut accumulator)?;
    }

    accumulator
        .ok_or_else(|| SKError::Conversion("streaming fit received no batches".into()))?
        .finalize()
}

/// Fold one batch into the accumulator, initializing it from the first batch.
fn accumulate_one(
    view: &SKCategoricalView<'_>,
    rows_count: usize,
    columns_count: usize,
    accumulator: &mut Option<StreamingCategories>,
) -> Result<(), SKError> {
    match view {
        SKCategoricalView::Nominal(matrix) => {
            let target = accumulator.get_or_insert_with(|| {
                StreamingCategories::Nominal(vec![Vec::new(); columns_count])
            });
            let StreamingCategories::Nominal(columns) = target else {
                return Err(representation_changed("nominal"));
            };
            for (column, labels) in columns.iter_mut().enumerate() {
                for row in 0..rows_count {
                    labels.push(matrix.value(row, column).to_owned());
                }
            }
        }
        SKCategoricalView::Integer(codes) => {
            let target = accumulator.get_or_insert_with(|| {
                StreamingCategories::Integer(vec![SKIntegerCategoryTable::default(); columns_count])
            });
            let StreamingCategories::Integer(tables) = target else {
                return Err(representation_changed("integer"));
            };
            for (column, table) in tables.iter_mut().enumerate() {
                for row in 0..rows_count {
                    let code = codes[[row, column]];
                    if code != SK_CATEGORICAL_MISSING_CODE {
                        table.observe(code);
                    }
                }
            }
        }
        _ => {
            return Err(representation_changed("unknown"));
        }
    }
    Ok(())
}

/// A representation changed between streaming batches.
fn representation_changed(representation: &'static str) -> SKError {
    SKError::UnsupportedRepresentation {
        representation,
        suggestion: "stream batches of a single categorical representation",
    }
}
