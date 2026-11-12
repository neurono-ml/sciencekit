//! Column-major binned feature store with category mapping (spec
//! `histogram-gradient-boosting`, design decisions 1 and 2).

use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::edges::SKNumericBinning;

/// Binning metadata of one feature: numeric quantile edges or observed
/// categorical codes.
#[derive(Debug, Clone, PartialEq)]
pub struct SKFeatureBinning<F> {
    numeric: Option<SKNumericBinning<F>>,
    categories: Vec<i64>,
}

impl<F: SKFloat> SKFeatureBinning<F> {
    fn numeric(numeric: SKNumericBinning<F>) -> Self {
        Self {
            numeric: Some(numeric),
            categories: Vec::new(),
        }
    }

    fn categorical(values: impl Iterator<Item = F>, maximum_bins: usize) -> Self {
        let mut categories: Vec<i64> = values
            .filter(|value| value.is_finite())
            .filter_map(|value| value.to_i64())
            .collect();
        categories.sort_unstable();
        categories.dedup();
        categories.truncate(maximum_bins.saturating_sub(1));
        Self {
            numeric: None,
            categories,
        }
    }

    /// Whether this feature is treated as categorical.
    pub fn is_categorical(&self) -> bool {
        self.numeric.is_none()
    }

    /// The interior quantile edges of a numeric feature (empty for
    /// categorical features).
    pub fn edges(&self) -> &[F] {
        self.numeric
            .as_ref()
            .map_or(&[], SKNumericBinning::edges)
    }

    /// The observed category codes of a categorical feature (empty for numeric
    /// features), ascending.
    pub fn categories(&self) -> &[i64] {
        &self.categories
    }

    /// The dedicated missing bin of a numeric feature.
    pub fn missing_bin(&self) -> Option<u8> {
        self.numeric.as_ref().map(SKNumericBinning::missing_bin)
    }

    /// The bin carrying categories never seen at fit time.
    pub fn unknown_bin(&self) -> u8 {
        self.categories.len() as u8
    }

    /// Number of bins that carry observed values (excluding the missing bin).
    pub fn number_of_bins(&self) -> usize {
        match &self.numeric {
            Some(numeric) => numeric.number_of_bins(),
            None => self.categories.len(),
        }
    }

    /// Assign a raw feature value to its bin.
    pub fn bin(&self, value: F) -> u8 {
        match &self.numeric {
            Some(numeric) => numeric.bin(value),
            None => {
                let code = value.to_i64().unwrap_or(i64::MAX);
                match self.categories.binary_search(&code) {
                    Ok(position) => position as u8,
                    Err(_) => self.unknown_bin(),
                }
            }
        }
    }
}

/// Column-major binned training design.
///
/// `bins` stores one `u8` per `(feature, row)` cell in feature blocks of
/// `number_of_rows`; a sparse input is binned from its stored entries plus the
/// implicit observed zero and is never densified.
#[derive(Debug, Clone, PartialEq)]
pub struct SKBinnedDataset<F> {
    number_of_rows: usize,
    number_of_features: usize,
    maximum_bins: usize,
    feature_binnings: Vec<SKFeatureBinning<F>>,
    bins: Vec<u8>,
}

impl<F: SKFloat> SKBinnedDataset<F> {
    /// Bin a dense or CSR design once, deriving one binning per feature.
    pub fn build(
        features: &SKDataView<'_, F>,
        categorical_features: &[usize],
        maximum_bins: usize,
    ) -> Result<Self, SKError> {
        let (number_of_rows, number_of_features) = match features {
            SKDataView::Dense(dense) => (dense.nrows(), dense.ncols()),
            SKDataView::Sparse(sparse) if sparse.is_csr() => (sparse.rows(), sparse.cols()),
            SKDataView::Sparse(_) => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR with to_csr",
                });
            }
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                });
            }
        };
        if let Some(&column) = categorical_features
            .iter()
            .find(|&&column| column >= number_of_features)
        {
            return Err(SKError::InvalidHyperparameter {
                name: "categorical_features",
                reason: format!(
                    "categorical column {column} is outside the fitted feature range 0..{number_of_features}"
                ),
            });
        }
        let feature_binnings: Vec<SKFeatureBinning<F>> = (0..number_of_features)
            .map(|feature| {
                let values = sk_feature_values(features, feature, number_of_rows);
                if categorical_features.contains(&feature) {
                    SKFeatureBinning::categorical(values.into_iter(), maximum_bins)
                } else {
                    SKFeatureBinning::numeric(SKNumericBinning::derive(
                        values.into_iter(),
                        maximum_bins,
                    ))
                }
            })
            .collect();
        let mut bins = vec![0u8; number_of_rows * number_of_features];
        feature_binnings
            .iter()
            .enumerate()
            .for_each(|(feature, binning)| {
                let block = &mut bins
                    [feature * number_of_rows..(feature + 1) * number_of_rows];
                match features {
                    SKDataView::Dense(dense) => {
                        block
                            .iter_mut()
                            .enumerate()
                            .for_each(|(row, bin)| *bin = binning.bin(dense[[row, feature]]));
                    }
                    SKDataView::Sparse(sparse) => {
                        block
                            .iter_mut()
                            .for_each(|bin| *bin = binning.bin(F::zero()));
                        (0..number_of_rows).for_each(|row| {
                            let sparse_row = sparse.outer_view(row).expect("row within bounds");
                            if let Ok(position) =
                                sparse_row.indices().binary_search(&feature)
                            {
                                block[row] = binning.bin(sparse_row.data()[position]);
                            }
                        });
                    }
                    _ => unreachable!("representation validated above"),
                }
            });
        Ok(Self {
            number_of_rows,
            number_of_features,
            maximum_bins,
            feature_binnings,
            bins,
        })
    }

    /// Number of binned rows.
    pub fn number_of_rows(&self) -> usize {
        self.number_of_rows
    }

    /// Number of binned features.
    pub fn number_of_features(&self) -> usize {
        self.number_of_features
    }

    /// The configured maximum bin count, also the index of the missing bin.
    pub fn maximum_bins(&self) -> usize {
        self.maximum_bins
    }

    /// The binning metadata of `feature_index`.
    pub fn feature_binning(&self, feature_index: usize) -> &SKFeatureBinning<F> {
        &self.feature_binnings[feature_index]
    }

    /// The binning metadata of every feature.
    pub fn feature_binnings(&self) -> &[SKFeatureBinning<F>] {
        &self.feature_binnings
    }

    /// The raw column-major bin storage.
    pub fn bins(&self) -> &[u8] {
        &self.bins
    }

    /// The bin block of `feature_index`.
    pub fn bins_for_feature(&self, feature_index: usize) -> &[u8] {
        &self.bins
            [feature_index * self.number_of_rows..(feature_index + 1) * self.number_of_rows]
    }

    /// The bin of `(feature_index, row_index)`.
    pub fn bin(&self, feature_index: usize, row_index: usize) -> u8 {
        self.bins[feature_index * self.number_of_rows + row_index]
    }
}

/// Collect the raw values observed for one feature: dense cells, or the stored
/// sparse entries plus the implicit observed zero of absent coordinates.
fn sk_feature_values<F: SKFloat>(
    features: &SKDataView<'_, F>,
    feature: usize,
    number_of_rows: usize,
) -> Vec<F> {
    match features {
        SKDataView::Dense(dense) => dense.column(feature).iter().copied().collect(),
        SKDataView::Sparse(sparse) => (0..number_of_rows)
            .map(|row| {
                let sparse_row = sparse.outer_view(row).expect("row within bounds");
                sparse_row
                    .indices()
                    .binary_search(&feature)
                    .map_or(F::zero(), |position| sparse_row.data()[position])
            })
            .collect(),
        _ => Vec::new(),
    }
}
