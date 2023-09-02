#![feature(core_intrinsics)]

use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::prelude::ThreadRng;
use rand::Rng;
use std::intrinsics::black_box;
use vers_vecs::{UDSTree, UDSTreeBuilder};

fn main() {
    let mut rng = rand::thread_rng();

    let (tree, nodes, number_children) = construct_random_tree(&mut rng, 1 << 20, 100);

    let sample = Uniform::new(0, number_children.len());
    let mut counter = 0;
    while counter < 100000000 {
        let mut i = sample.sample(&mut rng);
        while number_children[i] == 0 {
            i = sample.sample(&mut rng);
        }

        let node_index = nodes[i];
        let child_index = rng.gen_range(0..number_children[i]);

        black_box(tree.nth_child(node_index, child_index));
        counter += 1;
    }
}

/// Construct a random dfuds tree and return its meta information as well
///
/// # Returns
/// - The tree
/// - the node indices of the nodes in the tree
/// - the number of children of each node
pub fn construct_random_tree(
    rng: &mut ThreadRng,
    tree_size: usize,
    max_children: usize,
) -> (UDSTree, Vec<usize>, Vec<usize>) {
    let mut builder = UDSTreeBuilder::with_capacity(tree_size);
    let mut nodes = Vec::with_capacity(tree_size);
    let mut number_children = vec![0; tree_size];

    // attach max to root, so that the root has at least one child
    nodes.push(
        builder
            .visit_node(max_children)
            .expect("failed to visit node"),
    );
    number_children[0] = max_children;

    let mut index = 1;
    let mut children = max_children + 1;

    while children < tree_size - max_children {
        let n = rng.gen_range(0..=max_children);
        nodes.push(builder.visit_node(n).expect("failed to visit node"));
        number_children[index] = n;
        index += 1;
        children += n;
    }

    nodes.push(
        builder
            .visit_node(tree_size - children - 1)
            .expect("failed to visit node"),
    );
    number_children[index] = tree_size - children - 1;
    builder.visit_remaining_nodes();

    (
        builder.build().expect("failed to build tree"),
        nodes,
        number_children,
    )
}
