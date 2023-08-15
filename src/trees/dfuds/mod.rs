use crate::{BitVec, RsVec};

/// The bit used to encode an open parenthesis.
/// Do not change this value, it it is used without referencing the constant at multiple places.
const OPEN: u64 = 0;

/// The bit used to encode a closed parenthesis.
/// Do not change this value, it it is used without referencing the constant at multiple places.
const CLOSE: u64 = 1;

/// Block size of the min-max tree supporting the forward/backward search.
const MIN_MAX_BLOCK_SIZE: usize = 256;

#[allow(long_running_const_eval)]
const EXCESS_LOOKUP: [u16; 1 << 16] = calculate_excess_lookup();

const fn min_arg(a: i16, b: i16) -> i16 {
    a * ((a <= b) as i16) + b * ((a > b) as i16)
}

const fn max_arg(a: i16, b: i16) -> i16 {
    a * ((a >= b) as i16) + b * ((a < b) as i16)
}

const fn calculate_excess_lookup() -> [u16; 1 << 16] {
    let mut lookup = [0; 1 << 16];
    let mut pattern = 0;

    while pattern < lookup.len() {
        let mut excess = if (pattern & 1) == OPEN as usize {
            1i16
        } else {
            -1i16
        };
        let mut min_excess = excess;
        let mut max_excess = excess;
        let mut bit = 1;

        while bit < 16 {
            if pattern & (1 << bit) == OPEN as usize {
                excess += 1;
            } else {
                excess -= 1;
            }
            min_excess = min_arg(min_excess, excess);
            max_excess = max_arg(max_excess, excess);
            bit += 1;
        }
        lookup[pattern] = encode_limb_min_max(excess, min_excess, max_excess);
        pattern += 1;
    }
    lookup
}

/// Encodes the minimum, maximum and total excess within a block of 16 parentheses.
/// The encoding is as follows:
/// - Each number can have the range -16 to 16, which requires 6 bits to encode.
/// We store it with an offset of 16, so that we don't have to deal with dual encoding.
/// - The six least significant bits encode the total excess.
/// - The next five bits encode the minimum excess. Since the minimum excess can never be 32,
///  the most significant bit is always zero and need not be stored.
/// - The five most significant bits encode the maximum excess. Since the maximum excess is
/// 32 if and only if the total excess is 32, we need not store the most significant bit,
/// and instead reuse the most significant bit of the total excess for it.
/// This way we can encode all three values in 16 bits.
const fn encode_limb_min_max(total_excess: i16, min_excess: i16, max_excess: i16) -> u16 {
    (((max_excess + 16) & 0b11111) << 11 | (min_excess + 16) << 6 | (total_excess + 16) & 0b111111)
        as u16
}
const fn get_minimum_excess(encoding: u16) -> i16 {
    ((encoding >> 6) & 0b11111) as i16 - 16
}

const fn get_maximum_excess(encoding: u16) -> i16 {
    ((encoding & 0b100000) | ((encoding >> 11) & 0b11111)) as i16 - 16
}

const fn get_total_excess(encoding: u16) -> i16 {
    (encoding & 0b111111) as i16 - 16
}

/// A node in the min-max tree that supports forward and backward search.
#[derive(Clone, Copy, Debug, Default)]
struct MinMaxNode {
    min_excess: isize,
    max_excess: isize,
    total_excess: isize,
}

/// Succinct Depth-first unary degree sequence tree that requires 2n + o(n) bits of space
/// to represent a tree with n nodes.
#[derive(Clone, Debug)]
pub struct UDSTree {
    tree: RsVec,
    // The offset of the first leaf in the min-max tree.
    leaf_offset: usize,
    min_max: Vec<MinMaxNode>,
}

impl Default for UDSTree {
    fn default() -> Self {
        UDSTreeBuilder::default().build().unwrap()
    }
}

impl UDSTree {
    /// Calculate the excess of open parenthesis up and including the given index.
    /// This value is always positive, since DFUDS trees are made up of balanced parentheses expressions.
    ///
    /// # Parameters
    /// - `node`: The index of the tree representation to calculate the excess for. Does not
    /// coincide with the node index in the tree.
    #[must_use]
    fn excess(&self, index: usize) -> usize {
        self.tree.rank0(index + 1) - self.tree.rank1(index + 1)
    }

    /// Find the minimum index greater than `position` that has the excess `excess` compared to
    /// `position` - 1. This query is only intended for open parenthesis, where the excess immediately
    /// left of the position is lower. The query is not defined for closed parenthesis.
    #[must_use]
    fn fwd_search(&self, position: usize, excess: usize) -> usize {
        debug_assert!(
            self.tree.get_unchecked(position) == OPEN,
            "query only works for open parenthesis"
        );

        // search initial block
        if let Some(result) = self.fwd_search_within_block(position + 1, excess, 1) {
            return result;
        }

        let base_excess = self.excess(position);

        // search tree
        let mut current_node = self.leaf_offset + position / MIN_MAX_BLOCK_SIZE;

        // upwards min-max tree search: search sibling nodes of the current node to see if they
        // contain a matching excess
        // we assume that we find a suitable position in the tree, because the tree should be a
        // balanced parentheses expression
        while current_node > 0 {
            if current_node % 2 == 1 {
                // search right sibling
                // the position must exist, so if we are here, there must be a right sibling
                debug_assert!(current_node + 1 < self.min_max.len());
                if self.min_max[current_node + 1].min_excess <= base_excess as isize {
                    current_node += 1;
                    break;
                }
            }

            current_node = (current_node - 1) / 2; // move on to parent
        }

        // downwards min-max tree search: take the left-most child of the target node
        // until we reach a leaf
        loop {
            if current_node >= self.leaf_offset {
                // target node is a leaf
                break;
            }

            // search children
            let left_child = current_node * 2 + 1;
            if self.min_max[left_child].min_excess <= base_excess as isize {
                current_node = left_child;
            } else {
                current_node = left_child + 1;
            }
        }

        debug_assert!(
            current_node > self.leaf_offset,
            "forward tree search returned first leaf, which is not possible"
        );
        // search block specified by the min-max tree
        self.fwd_search_within_block(
            current_node * MIN_MAX_BLOCK_SIZE,
            base_excess,
            self.min_max[current_node - 1].total_excess, // this cannot fail, because we can't possibly be in the first leaf
        )
        .expect("The min-max tree confirmed a matching excess, but the block does not contain it")
    }

    /// Search a block of bits within the bitvector for an index right of `position`
    /// with the given `excess`. The excess at `position` is given by `current_excess`.
    fn fwd_search_within_block(
        &self,
        position: usize,
        excess: usize,
        mut current_excess: isize,
    ) -> Option<usize> {
        let mut index = position;

        // search within the current block until a position aligned to the lookup table is reached
        while index % 16 != 0 {
            if index / MIN_MAX_BLOCK_SIZE > position / MIN_MAX_BLOCK_SIZE {
                return None;
            }

            current_excess += if self.tree.get_unchecked(index) == OPEN {
                1
            } else {
                -1
            };

            if current_excess == excess as isize {
                return Some(index);
            }

            index += 1;
        }

        // skip limbs if they don't contain the target excess. Assume that the current excess is
        // higher than the target excess, because we are dealing with balanced parentheses
        // expressions.
        while index / MIN_MAX_BLOCK_SIZE == position / MIN_MAX_BLOCK_SIZE
            && index + 16 < self.tree.len()
        {
            let lookup = EXCESS_LOOKUP[self.tree.get_bits_unchecked(index, 16) as usize];
            let min_excess = get_minimum_excess(lookup) as isize;

            if current_excess + min_excess > excess as isize {
                index += 16;
                current_excess += get_total_excess(lookup) as isize;
            } else {
                for _ in 0..16 {
                    current_excess += if self.tree.get_unchecked(index) == OPEN {
                        1
                    } else {
                        -1
                    };
                    if current_excess == excess as isize {
                        return Some(index);
                    }
                    index += 1;
                }
                unreachable!("Did not find target excess in limb even though it should be there");
            }
        }

        // if there is a non-full limb left, search it bit by bit. This means we are in the last
        // block of the tree, so we can't possibly fail
        while index < self.tree.len() {
            current_excess += if self.tree.get_unchecked(index) == OPEN {
                1
            } else {
                -1
            };

            if current_excess == excess as isize {
                return Some(index);
            }

            index += 1;
        }

        return None;
    }

    /// Find the maximum index smaller than `position` such that `index - 1` has the excess `excess` compared to
    /// `position`.
    #[must_use]
    fn bwd_search(&self, position: usize, excess: usize) -> usize {
        // search initial block
        if let Some(result) = self.bwd_search_within_block(position - 1, excess, if self.tree.get_unchecked(position) == CLOSE { 1 } else { -1 }) {
            return result;
        }

        let base_excess = self.excess(position);

        // search tree
        let mut current_node = self.leaf_offset + position / MIN_MAX_BLOCK_SIZE;

        // upwards min-max tree search: search sibling nodes of the current node to see if they
        // contain a matching excess
        // we assume that we find a suitable position in the tree, because the tree should be a
        // balanced parentheses expression
        while current_node > 0 {
            if current_node % 2 == 0 {
                // search left sibling
                // the position must exist, so if we are here, there must be a left sibling
                debug_assert!(current_node - 1 < self.min_max.len());
                if self.min_max[current_node - 1].min_excess <= base_excess as isize || self.min_max[current_node - 1].max_excess >= base_excess as isize {
                    current_node -= 1;
                    break;
                }
            }

            current_node = (current_node - 1) / 2; // move on to parent
        }

        // downwards min-max tree search: take the right-most child of the target node
        // until we reach a leaf
        loop {
            if current_node >= self.leaf_offset {
                // target node is a leaf
                break;
            }

            // search children
            let right_child = current_node * 2 + 2;
            if self.min_max[right_child].min_excess <= base_excess as isize || self.min_max[right_child].max_excess >= base_excess as isize {
                current_node = right_child;
            } else {
                current_node = right_child - 1;
            }
        }

        debug_assert!(
            current_node < self.min_max.len() - 1,
            "backward tree search returned last leaf, which is not possible"
        );
        // search block specified by the min-max tree
        self.bwd_search_within_block(
            (current_node - self.leaf_offset + 1) * MIN_MAX_BLOCK_SIZE - 1,
            base_excess,
            self.min_max[current_node].total_excess,
        )
            .expect("The min-max tree confirmed a matching excess, but the block does not contain it")
    }

    /// Search a block of bits within the bitvector for an index right of `position`
    /// with the given `excess`. The excess at `position` is given by `current_excess`.
    fn bwd_search_within_block(
        &self,
        position: usize,
        excess: usize,
        mut current_excess: isize,
    ) -> Option<usize> {
        let mut index = position;

        // search within the current block until a position aligned to the lookup table is reached
        while index % 16 != 15 {
            if index / MIN_MAX_BLOCK_SIZE < position / MIN_MAX_BLOCK_SIZE {
                return None;
            }

            current_excess += if self.tree.get_unchecked(index) == CLOSE {
                1
            } else {
                -1
            };

            if -current_excess == excess as isize {
                return Some(index);
            }

            index -= 1;
        }

        // skip limbs if they don't contain the target excess. Assume that the current excess is
        // higher than the target excess, because we are dealing with balanced parentheses
        // expressions.
        while index / MIN_MAX_BLOCK_SIZE == position / MIN_MAX_BLOCK_SIZE && index >= 15
        {
            let lookup = EXCESS_LOOKUP[self.tree.get_bits_unchecked(index - 15, 16) as usize];
            let min_excess = get_minimum_excess(lookup) as isize;
            let max_excess = get_maximum_excess(lookup) as isize;
            let total_excess = get_total_excess(lookup) as isize;

            if current_excess - total_excess + min_excess > excess as isize || current_excess - total_excess + max_excess < excess as isize {
                current_excess -= total_excess;

                // check the last member of the limb, because we won't check it in the loop
                if -current_excess == excess as isize {
                    return Some(index + 1 - 16)
                }
                index -= 16;
            } else {
                for _ in 0..16 {
                    current_excess += if self.tree.get_unchecked(index) == CLOSE {
                        1
                    } else {
                        -1
                    };
                    if -current_excess == excess as isize {
                        return Some(index);
                    }
                    index -= 1;
                }
                unreachable!("Did not find target excess in limb even though it should be there");
            }
        }

        return None;
    }

    /// Find a matching closing parenthesis for the parenthesis at `position`. Assumes that the
    /// parenthesis at `position` is an open parenthesis. The query is not defined for closed
    /// parenthesis.
    /// Returns the index of the closing parenthesis.
    #[must_use]
    fn find_close(&self, position: usize) -> usize {
        self.fwd_search(position, 0)
    }

    /// Find a matching opening parenthesis for the parenthesis at `position`. Assumes that the
    /// parenthesis at `position` is a closed parenthesis. The query is not defined for open
    /// parenthesis.
    /// Returns the index of the opening parenthesis.
    #[must_use]
    fn find_open(&self, position: usize) -> usize {
        debug_assert!(
            self.tree.get_unchecked(position) == CLOSE,
            "query only works for closed parenthesis"
        );
        self.bwd_search(position, 0)
    }

    /// Returns the number of nodes in the tree. Since an empty tree is not allowed, the number of
    /// nodes is always greater than zero.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tree.len() / 2
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

    /// Creates a new builder with the given node capacity pre-allocated (including the root node).
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

        self.tree
            .append_bits(CLOSE << (children % 64), (children % 64) + 1);
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

        // generate min-max tree in a linear vector
        let blocks = (self.tree.len() + (MIN_MAX_BLOCK_SIZE - 1)) / MIN_MAX_BLOCK_SIZE;
        let tree_depth = (blocks as f64).log2().ceil() as usize;

        // if the vector only fills one block, we don't need a min-max tree
        if tree_depth == 0 {
            return Ok(UDSTree {
                tree: RsVec::from_bit_vec(self.tree),
                leaf_offset: 0,
                min_max: vec![],
            });
        }

        // TODO: do not use sentinel values, but actually cut out the unused nodes
        let mut min_max_tree = vec![
            MinMaxNode {
                min_excess: isize::MAX,
                max_excess: isize::MIN,
                total_excess: self.tree.len() as isize
            };
            (1 << (tree_depth + 1)) - 1
        ];

        // total excess across the entire dfuds tree
        let mut excess = 0;

        // initialize leaf nodes
        let leaf_offset = (1 << tree_depth) - 1;
        min_max_tree[leaf_offset..]
            .iter_mut()
            .enumerate()
            .for_each(|(i, node)| {
                debug_assert!(
                    MIN_MAX_BLOCK_SIZE % 64 == 0,
                    "block cannot be retrieved as limbs"
                );

                // TODO we should prune the unused leafs instead of aborting every single read here
                // this is a hack to make sure we don't read past the end of the vector
                if i * MIN_MAX_BLOCK_SIZE < self.tree.len() {
                    // min excess is stored per-block
                    let mut min = excess
                        + if self.tree.get_unchecked(i * MIN_MAX_BLOCK_SIZE) == OPEN {
                            1
                        } else {
                            -1
                        };

                    let mut max = min;

                    for j in 0..MIN_MAX_BLOCK_SIZE / 64 {
                        let mut length = 64;
                        if i * MIN_MAX_BLOCK_SIZE + (j + 1) * 64 >= self.tree.len() {
                            length = self.tree.len() % 64;
                        };

                        let limb = self
                            .tree
                            .get_bits_unchecked(i * MIN_MAX_BLOCK_SIZE + j * 64, length);

                        // update min excess
                        for k in 0..length {
                            let bit = limb & (1 << k);
                            if bit == OPEN {
                                excess += 1;
                            } else {
                                excess -= 1;
                            }

                            min = min.min(excess);
                            max = max.max(excess);
                        }

                        if length < 64 {
                            break;
                        }
                    }

                    *node = MinMaxNode {
                        min_excess: min,
                        max_excess: max,
                        total_excess: excess,
                    };
                }
            });

        for depth in (0..tree_depth).rev() {
            for i in ((1 << depth) - 1)..((2 << depth) - 1) {
                let left = min_max_tree[2 * i + 1];
                let right = min_max_tree[2 * i + 2];
                min_max_tree[i] = MinMaxNode {
                    min_excess: left.min_excess.min(right.min_excess),
                    max_excess: left.max_excess.max(right.max_excess),
                    total_excess: right.total_excess,
                };
            }
        }

        Ok(UDSTree {
            tree: RsVec::from_bit_vec(self.tree),
            leaf_offset,
            min_max: min_max_tree,
        })
    }
}

#[cfg(test)]
mod tests;
