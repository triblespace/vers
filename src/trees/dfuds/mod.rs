use crate::{BitVec, RsVec};

const OPEN: u64 = 0;
const CLOSE: u64 = 1;

/// Succinct Depth-first unary degree sequence tree that requires 2n + o(n) bits of space
/// to represent a tree with n nodes.
#[derive(Clone, Debug)]
pub struct UDSTree {
    tree: RsVec,
}

impl Default for UDSTree {
    fn default() -> Self {
        UDSTreeBuilder::default().build().unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct UDSTreeBuilder {
    tree: BitVec,
    balance: usize,
}

impl Default for UDSTreeBuilder {
    fn default() -> Self {
        UDSTreeBuilder::with_capacity(32)
    }
}

impl UDSTreeBuilder {
    /// Creates a new empty builder.
    /// The builder can be used to create a tree by appending children to the current node.
    /// The root node must be visited, an empty tree cannot be created.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new builder with the given node capacity pre-allocated.
    #[must_use]
    pub fn with_capacity(nodes: usize) -> Self {
        let mut bit_vec = BitVec::with_capacity(nodes * 2);
        bit_vec.append_bit(OPEN);

        Self {
            tree: bit_vec,
            balance: 1,
        }
    }

    /// Appends children to the current node.
    /// The current node is the next node in depth-first order that has not been visited yet.
    /// The children are appended in depth-first order (whatever that means for the client).
    /// The function must be called for each node in the tree once.
    /// The function must be called at least once, because each tree must have at least one root
    /// node.
    /// The function must not be called after all nodes have been visited, i.e. for each call to
    /// `visit_node`, a child must have been appended beforehand.
    /// Calling the function after all nodes have been visited will return an error.
    ///
    /// # Parameters
    /// - `children`: The number of children of the current node.
    ///
    /// # Examples
    /// ```rust
    /// use vers_vecs::trees::dfuds::UDSTreeBuilder;
    ///
    /// // To create a tree of this shape:
    /// //     0
    /// //    / \
    /// //   1   2
    /// //  / \   \
    /// // 3   4   5
    /// // We visit the nodes in depth first order (0, 1, 3, 4, 2, 5) and append the number of
    /// // children of each node:
    /// let mut builder = UDSTreeBuilder::with_capacity(6);
    /// builder.visit_node(2)?; // root node (0)
    /// builder.visit_node(2)?; // node 1
    /// builder.visit_node(0)?; // node 3
    /// builder.visit_node(0)?; // node 4
    /// builder.visit_node(1)?; // node 2
    /// builder.visit_node(0)?; // node 5
    ///
    /// // attempting to visit another node will fail, because we already visited all nodes:
    /// assert!(builder.visit_node(0).is_err());
    ///
    /// // The tree can be built from the builder. Attempting to build the tree before visiting
    /// // all nodes will fail.
    /// let tree = builder.build()?;
    /// # Ok::<(), String>(())
    /// ```
    pub fn visit_node(&mut self, children: usize) -> Result<(), String> {
        if self.balance == 0 {
            return Err("Tree is already complete".to_string());
        }

        // append `children` zeros (open parentheses) and one one (closed parentheses)
        for _ in 0..children / 64 {
            self.tree.append_word(0);
            self.balance += 64;
        }

        self.tree.append_bits(CLOSE << (children % 64), (children % 64) + 1);
        self.balance += children % 64;
        self.balance -= 1;
        Ok(())
    }

    /// Visits the remaining nodes in the tree and closes them without appending any children.
    /// This function is an alternative to calling `visit_node` with `0` as the number of children
    /// for each remaining node.
    ///
    /// # Examples
    /// ```rust
    /// use vers_vecs::trees::dfuds::UDSTreeBuilder;
    ///
    /// // To create a tree of this shape:
    /// //     0
    /// //    /|\
    /// //   1 2 3
    /// //  /|
    /// // 4 5
    /// // We visit the nodes in depth first order (0, 1, 4, 5, 2, 3) and append the number of
    /// // children of each node, but we can skip visiting 4, 5, 2, and 3, because they have no
    /// // children and there are no more nodes with children after any of them:
    ///
    /// let mut builder = UDSTreeBuilder::with_capacity(6);
    /// builder.visit_node(3)?; // root node (0)
    /// builder.visit_node(2)?; // node 1
    /// builder.visit_remaining_nodes(); // finishes all open nodes assuming they have no children
    ///
    /// // This always finishes the tree, so we can build it:
    /// let tree = builder.build()
    ///     .expect("we know the tree is complete because of visit_remaining_nodes");
    /// # Ok::<(), String>(())
    /// ```
    pub fn visit_remaining_nodes(&mut self) {
        for _ in 0..self.balance / 64 {
            self.tree.append_word(u64::MAX);
        }
        if self.balance % 64 > 0 {
            self.tree.append_bits(u64::MAX, self.balance % 64);
        }
        self.balance = 0;
    }

    /// Builds the tree from the builder.
    /// Attempting to build the tree before visiting all nodes will fail.
    ///
    /// # Examples
    /// ```rust
    /// use vers_vecs::trees::dfuds::UDSTreeBuilder;
    ///
    /// // Create a tree of this shape:
    /// //     0
    /// //    / \
    /// //   1   2
    ///
    /// let mut builder = UDSTreeBuilder::with_capacity(3);
    /// builder.visit_node(2)?; // root node (0)
    /// builder.visit_remaining_nodes(); // finishes all open nodes assuming they have no children
    ///
    /// assert!(builder.build().is_ok());
    /// # Ok::<(), String>(())
    /// ```
    ///
    /// If not all nodes have been visited, building the tree will fail:
    /// ```rust
    /// use vers_vecs::trees::dfuds::UDSTreeBuilder;
    ///
    /// // append 4 children to the root node, but only visit 3 of them:
    /// let mut builder = UDSTreeBuilder::with_capacity(5);
    /// builder.visit_node(4)?; // root node (0)
    /// builder.visit_node(0)?; // node 1
    /// builder.visit_node(0)?; // node 2
    /// builder.visit_node(0)?; // node 3
    ///
    /// // attempting to build the tree will fail, because we did not visit all nodes:
    /// assert!(builder.build().is_err());
    /// # Ok::<(), String>(())
    /// ```
    pub fn build(self) -> Result<UDSTree, String> {
        if self.balance > 0 {
            return Err(format!("Tree has {} unvisited nodes", self.balance));
        }

        Ok(UDSTree {
            tree: RsVec::from_bit_vec(self.tree),
        })
    }
}