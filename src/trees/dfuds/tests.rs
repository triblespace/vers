use crate::trees::dfuds::{UDSTree, UDSTreeBuilder, CLOSE, MIN_MAX_BLOCK_SIZE};
use crate::{BitVec, RsVec};
use rand::Rng;

#[test]
fn test_fwd_search_within_block() {
    let mut expr = BitVec::from_zeros(10);
    // generate expression: ((()())())
    expr.flip_bit(3);
    expr.flip_bit(5);
    expr.flip_bit(6);
    expr.flip_bit(8);
    expr.flip_bit(9);

    let tree = UDSTree {
        tree: RsVec::from_bit_vec(expr),
        leaf_offset: 0,  // mock data, not used
        min_max: vec![], // mock data, not used
    };

    assert_eq!(tree.fwd_search(1, 0), 6);
    assert_eq!(tree.fwd_search(0, 0), 9);
    assert_eq!(tree.fwd_search(0, 1), 6);
    assert_eq!(tree.fwd_search(2, 0), 3);
}

#[test]
fn test_fwd_search_within_full_block() {
    let mut expr = BitVec::from_zeros(MIN_MAX_BLOCK_SIZE);
    for i in MIN_MAX_BLOCK_SIZE / 2..MIN_MAX_BLOCK_SIZE {
        expr.flip_bit(i);
    }

    let tree = UDSTree {
        tree: RsVec::from_bit_vec(expr),
        leaf_offset: 0,  // mock data, not used
        min_max: vec![], // mock data, not used
    };

    for i in 0..MIN_MAX_BLOCK_SIZE / 2 {
        assert_eq!(
            tree.fwd_search(i, 0),
            MIN_MAX_BLOCK_SIZE - i - 1,
            "fwd_search({}, 0) failed",
            i
        );
    }
}

#[test]
fn construct_tree() {
    // build empty tree
    let mut builder = UDSTreeBuilder::new();
    builder.visit_remaining_nodes();
    assert!(builder.build().is_ok());

    // minimal tree that doesn't require a min-max tree
    let mut builder = UDSTreeBuilder::new();
    assert!(builder.visit_node(64).is_ok());
    builder.visit_remaining_nodes();
    assert!(builder.build().is_ok());

    // tree with 2 blocks
    let mut builder = UDSTreeBuilder::new();
    assert!(builder.visit_node(130).is_ok());
    builder.visit_remaining_nodes();
    assert!(builder.build().is_ok());

    // tree with many blocks
    let mut builder = UDSTreeBuilder::new();
    for _ in 0..20 {
        assert!(builder.visit_node(128).is_ok());
    }
    builder.visit_remaining_nodes();
    assert!(builder.build().is_ok());
}

#[test]
fn test_fwd_search_across_tree() {
    let mut tree = UDSTreeBuilder::with_capacity(512);
    tree.visit_node(500)
        .expect("failed to visit node with 500 children");
    tree.visit_remaining_nodes();
    let tree = tree.build().expect("failed to build tree");

    assert_eq!(tree.fwd_search(0, 0), 1001);
    assert_eq!(tree.fwd_search(1, 0), 1000);
    assert_eq!(tree.fwd_search(500, 0), 501);

    assert_eq!(tree.fwd_search(0, 2), 1);
    assert_eq!(tree.fwd_search(3, 3), 5);
}

#[test]
fn test_randomized_find_close() {
    const TREE_SIZE: usize = 2000;
    const MAX_CHILDREN: usize = 20;

    let mut tree = UDSTreeBuilder::with_capacity(TREE_SIZE);
    let mut children = 0;
    let mut rng = rand::thread_rng();

    // generate random tree
    while children < (TREE_SIZE - MAX_CHILDREN) {
        let n = rng.gen_range(1..MAX_CHILDREN);
        tree.visit_node(n).expect("failed to append children");
        children += n;
    }
    tree.visit_node(TREE_SIZE - children - 1)
        .expect("failed to append last children");
    tree.visit_remaining_nodes();
    let tree = tree.build().expect("failed to build tree");

    // for each open bracket, find the matching close bracket, and check if the close bracket's excess matches
    // and if the close bracket is the first close bracket with that excess
    for i in 0..tree.tree.len() {
        if tree.tree.get_unchecked(i) == CLOSE {
            continue;
        }

        let excess = tree.excess(i);
        let close = tree.find_close(i);
        assert_eq!(tree.excess(close), excess - 1);
        assert_eq!(
            close,
            tree.tree
                .iter()
                .enumerate()
                .skip(i)
                .find(|&(i, b)| b == CLOSE && tree.excess(i) == excess - 1)
                .map(|(i, _)| i)
                .unwrap()
        );
    }
}
