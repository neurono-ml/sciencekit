//! KD-tree node storage backed by contiguous permutation ranges.

/// One immutable KD-tree node.
///
/// `start` and `end` delimit the node's contiguous `[start, end)` range in the
/// tree's permutation. Internal nodes carry the indices of their two children;
/// leaves have no children. Every node stores the axis-aligned box enclosing its
/// members for pruning.
#[derive(Debug, Clone)]
pub struct SKKDTreeNode<F> {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) left: Option<usize>,
    pub(crate) right: Option<usize>,
    pub(crate) minimum: Vec<F>,
    pub(crate) maximum: Vec<F>,
}

impl<F> SKKDTreeNode<F> {
    /// A node without children covering `[start, end)`.
    pub(crate) fn new(
        start: usize,
        end: usize,
        minimum: Vec<F>,
        maximum: Vec<F>,
    ) -> SKKDTreeNode<F> {
        SKKDTreeNode {
            start,
            end,
            left: None,
            right: None,
            minimum,
            maximum,
        }
    }

    /// Whether this node is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// The first permutation position covered by this node.
    pub fn start(&self) -> usize {
        self.start
    }

    /// One past the last permutation position covered by this node.
    pub fn end(&self) -> usize {
        self.end
    }

    /// The left child index, when this node has children.
    pub fn left(&self) -> Option<usize> {
        self.left
    }

    /// The right child index, when this node has children.
    pub fn right(&self) -> Option<usize> {
        self.right
    }

    /// The per-feature lower corner of the enclosing box.
    pub fn minimum(&self) -> &[F] {
        &self.minimum
    }

    /// The per-feature upper corner of the enclosing box.
    pub fn maximum(&self) -> &[F] {
        &self.maximum
    }
}
