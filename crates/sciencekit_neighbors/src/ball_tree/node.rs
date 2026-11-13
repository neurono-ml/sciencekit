//! Ball-tree node storage backed by contiguous permutation ranges.

/// One immutable ball-tree node.
///
/// `start` and `end` delimit the node's contiguous `[start, end)` range in the
/// tree's permutation. Internal nodes carry the indices of their two children;
/// leaves have no children. Every node stores the centroid and the Euclidean
/// covering radius of its members.
#[derive(Debug, Clone)]
pub struct SKBallTreeNode<F> {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) left: Option<usize>,
    pub(crate) right: Option<usize>,
    pub(crate) centroid: Vec<F>,
    pub(crate) radius: F,
}

impl<F> SKBallTreeNode<F> {
    /// A node without children covering `[start, end)`.
    pub(crate) fn new(start: usize, end: usize, centroid: Vec<F>, radius: F) -> SKBallTreeNode<F> {
        SKBallTreeNode {
            start,
            end,
            left: None,
            right: None,
            centroid,
            radius,
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

    /// The centroid of the node's members.
    pub fn centroid(&self) -> &[F] {
        &self.centroid
    }

    /// The Euclidean covering radius of the node's members.
    pub fn radius(&self) -> F
    where
        F: Copy,
    {
        self.radius
    }
}
