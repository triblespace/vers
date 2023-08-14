use crate::trees::dfuds::{UDSTree, UDSTreeBuilder, MIN_MAX_BLOCK_SIZE};
use crate::{BitVec, RsVec};

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
