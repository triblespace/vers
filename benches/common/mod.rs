#![allow(dead_code)]

use criterion::PlotConfiguration;
use indextree::{Arena, NodeEdge, NodeId};
use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use vers_vecs::{BitVec, RsVec, UDSTree, UDSTreeBuilder};

pub const SIZES: [usize; 13] = [
    1 << 8,
    1 << 10,
    1 << 12,
    1 << 14,
    1 << 16,
    1 << 18,
    1 << 20,
    1 << 22,
    1 << 24,
    1 << 26,
    1 << 28,
    1 << 30,
    1 << 32,
];

pub const LIMITED_SIZES: [usize; 10] = [
    1 << 8,
    1 << 10,
    1 << 12,
    1 << 14,
    1 << 16,
    1 << 18,
    1 << 20,
    1 << 22,
    1 << 24,
    1 << 26,
];

pub fn construct_vers_vec(rng: &mut ThreadRng, len: usize) -> RsVec {
    let sample = Uniform::new(0, u64::MAX);

    let mut bit_vec = BitVec::new();
    for _ in 0..len / 64 {
        bit_vec.append_word(sample.sample(rng));
    }

    RsVec::from_bit_vec(bit_vec)
}

pub fn fill_random_vec(rng: &mut ThreadRng, len: usize) -> Vec<u64> {
    let sample = Uniform::new(0, u64::MAX);

    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(sample.sample(rng));
    }

    vec
}

/// Construct a random dfuds tree and return its meta information as well
///
/// # Returns
/// - The tree
/// - the node indices of the nodes in the tree
pub fn construct_random_tree(
    rng: &mut ThreadRng,
    tree_size: usize,
) -> (UDSTree, Vec<usize>, Vec<usize>) {
    let (index_tree, root) = construct_random_index_tree(rng, tree_size);

    let mut builder = UDSTreeBuilder::with_capacity(tree_size);
    let mut nodes = Vec::with_capacity(tree_size);
    let mut number_children = vec![0; tree_size];

    let mut index = 0;

    root.traverse(&index_tree).for_each(|edge| match edge {
        NodeEdge::Start(node_id) => {
            let children = node_id.children(&index_tree).count();
            let node = builder.visit_node(children).expect("failed to visit node");
            nodes.push(node);
            number_children[index] = children;
            index += 1;
        }
        NodeEdge::End(_) => {}
    });

    (
        builder.build().expect("failed to build tree"),
        nodes,
        number_children,
    )
}

fn construct_random_index_tree(rng: &mut ThreadRng, tree_size: usize) -> (Arena<()>, NodeId) {
    // generate prüfer sequence
    let mut prufer = Vec::with_capacity(tree_size - 2);
    let sample = Uniform::new(0, tree_size);
    for _ in 0..tree_size - 2 {
        prufer.push(sample.sample(rng));
    }

    // generate degree sequence
    let mut degree_sequence = vec![1; tree_size];
    for i in &prufer {
        degree_sequence[*i] += 1;
    }

    // prepare index tree
    let mut arena = Arena::new();
    let mut nodes = Vec::with_capacity(tree_size);
    for _ in 0..tree_size {
        nodes.push(arena.new_node(()));
    }

    // prepare priority queue
    let mut priority_queue = BinaryHeap::new();
    for j in 0..tree_size {
        if degree_sequence[j] == 1 {
            priority_queue.push(Reverse(j));
        }
    }

    for &i in &prufer {
        let j = priority_queue.pop().unwrap().0;

        nodes[i].append(nodes[j], &mut arena);
        degree_sequence[i] -= 1;
        degree_sequence[j] -= 1;

        if degree_sequence[i] == 1 {
            priority_queue.push(Reverse(i));
        }
    }

    // find remaining degree 1 nodes:
    assert_eq!(
        priority_queue.len(),
        2,
        "expected 2 remaining nodes, found {}",
        priority_queue.len()
    );

    // connect remaining nodes
    nodes[priority_queue.pop().unwrap().0]
        .append(nodes[priority_queue.pop().unwrap().0], &mut arena);

    let root = nodes[0].ancestors(&arena).last().unwrap();
    (arena, root)
}

pub fn plot_config() -> PlotConfiguration {
    PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic)
}
