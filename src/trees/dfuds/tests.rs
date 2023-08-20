use crate::trees::dfuds::{UDSTree, UDSTreeBuilder, CLOSE, MIN_MAX_BLOCK_SIZE, OPEN};
use crate::{BitVec, RsVec};
use rand::Rng;

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
fn test_bwd_search_within_block() {
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

    assert_eq!(tree.bwd_search(9, 0), 0);
    assert_eq!(tree.bwd_search(8, 0), 7);
    assert_eq!(tree.bwd_search(7, -2), 0);
    assert_eq!(tree.bwd_search(6, 0), 1);
}

#[test]
fn test_bwd_search_with_lookup() {
    let mut expr = BitVec::from_zeros(64);
    for i in 32..64 {
        expr.flip_bit(i);
    }

    let tree = UDSTree {
        tree: RsVec::from_bit_vec(expr),
        leaf_offset: 0,  // mock data, not used
        min_max: vec![], // mock data, not used
    };

    for i in 0..32 {
        assert_eq!(
            tree.bwd_search(64 - i - 1, 0),
            i,
            "bwd_search({}, 0) failed",
            MIN_MAX_BLOCK_SIZE - i - 1
        );
    }
}

#[test]
fn test_bwd_search_within_full_block() {
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
            tree.bwd_search(MIN_MAX_BLOCK_SIZE - i - 1, 0),
            i,
            "bwd_search({}, 0) failed",
            MIN_MAX_BLOCK_SIZE - i - 1
        );
    }
}

#[test]
fn test_bwd_search_across_tree() {
    let mut tree = UDSTreeBuilder::with_capacity(512);
    tree.visit_node(500)
        .expect("failed to visit node with 500 children");
    tree.visit_remaining_nodes();
    let tree = tree.build().expect("failed to build tree");

    assert_eq!(tree.bwd_search(1001, 0), 0);
    assert_eq!(tree.bwd_search(1000, 0), 1);
    assert_eq!(tree.bwd_search(501, 0), 500);

    assert_eq!(tree.bwd_search(450, -2), 449);
    assert_eq!(tree.bwd_search(3, -2), 2);
}

fn generate_random_tree() -> UDSTree {
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
    tree.build().expect("failed to build tree")
}

#[test]
fn test_randomized_find_close() {
    let tree = generate_random_tree();

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

#[test]
fn test_randomized_find_open() {
    let tree = generate_random_tree();

    // for each open bracket, find the matching close bracket, and check if the close bracket's excess matches
    // and if the close bracket is the first close bracket with that excess
    for i in 0..tree.tree.len() {
        if tree.tree.get_unchecked(i) == OPEN {
            continue;
        }

        let excess = tree.excess(i);
        let open = tree.find_open(i);
        assert_eq!(tree.excess(open), excess + 1);
        assert_eq!(
            open,
            tree.tree
                .iter()
                .enumerate()
                .rev()
                .skip(tree.tree.len() - 1 - i)
                .find(|&(i, b)| b == OPEN && tree.excess(i) == excess + 1)
                .map(|(i, _)| i)
                .unwrap()
        );
    }
}

#[test]
fn test_degree() {
    const TREE_SIZE: usize = 2000;

    let mut tree = UDSTreeBuilder::with_capacity(TREE_SIZE);
    let mut rng = rand::thread_rng();
    let mut nodes = Vec::with_capacity(4);
    let mut degrees = Vec::with_capacity(4);

    // append 4 children to root
    let root_node = tree
        .visit_node(4)
        .expect("failed to append children to root");

    // generate random amount of children:
    for _ in 0..4 {
        let n = rng.gen_range(1..TREE_SIZE / 4);
        degrees.push(n);
        nodes.push(tree.visit_node(n).expect("failed to append children"));
        for _ in 0..n {
            tree.visit_node(0).expect("failed to close children");
        }
    }

    let tree = tree.build().expect("failed to build tree");

    // root
    assert_eq!(tree.degree(root_node), 4);

    // children
    for i in 0..4 {
        assert_eq!(tree.degree(nodes[i]), degrees[i]);
    }
}

#[test]
fn test_nth_child() {
    let mut tree = UDSTreeBuilder::with_capacity(882);

    let root = tree.visit_node(20).expect("failed to append children");
    for _ in 0..12 {
        tree.visit_node(0).expect("failed to close children");
    }
    let child_thirteen = tree.visit_node(20).expect("failed to append children");
    let child_zero = tree.visit_node(400).expect("failed to append children");
    for _ in 0..300 {
        tree.visit_node(0).expect("failed to close children");
    }
    let last_child = tree.visit_node(0).expect("failed to append children");
    tree.visit_remaining_nodes();
    let tree = tree.build().expect("failed to build tree");

    assert_eq!(tree.nth_child(root, 12), child_thirteen);
    assert_eq!(tree.nth_child(child_thirteen, 0), child_zero);
    assert_eq!(tree.nth_child(child_zero, 300), last_child);
}

#[test]
fn test_nth_child_fully() {
    const CHILDREN: [usize; 23] = [4, 2, 3, 0, 0, 0, 6, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 1, 0, 1, 0];
    let mut nodes = Vec::with_capacity(CHILDREN.len());

    let mut tree = UDSTreeBuilder::new();
    for c in CHILDREN {
        nodes.push(tree.visit_node(c as usize).expect("failed to append children"));
    }
    let tree = tree.build().expect("failed to build tree");

    for (i, &c) in CHILDREN.iter().enumerate() {
        if c > 0 {
            for j in 0..c {
                // simulate DFS to determine child index
                let mut stack = j;
                let mut cursor = i + 1;
                while stack > 0 {
                    stack += CHILDREN[cursor];
                    cursor += 1;
                    stack -= 1;
                }

                assert_eq!(tree.nth_child(nodes[i], j), nodes[cursor], "failed to retrieve {}. child of {}. node (idx: {}) (which has {} children)", j, i, nodes[i], c);

            }
        }
    }

}

#[test]
fn test_parent() {
    let mut tree = UDSTreeBuilder::with_capacity(882);

    let root = tree.visit_node(20).expect("failed to append children");
    for _ in 0..12 {
        tree.visit_node(0).expect("failed to close children");
    }
    let child_thirteen = tree.visit_node(20).expect("failed to append children");
    let child_zero = tree.visit_node(400).expect("failed to append children");
    for _ in 0..300 {
        tree.visit_node(0).expect("failed to close children");
    }
    let last_child = tree.visit_node(0).expect("failed to append children");
    tree.visit_remaining_nodes();
    let tree = tree.build().expect("failed to build tree");

    assert_eq!(tree.parent(child_thirteen), root);
    assert_eq!(tree.parent(child_zero), child_thirteen);
    assert_eq!(tree.parent(last_child), child_zero);
    assert_eq!(tree.parent(880), root);
}

#[test]
fn test_subtree_size() {
    let mut tree = UDSTreeBuilder::with_capacity(1211);

    let root = tree.visit_node(2).expect("failed to append children");

    // subtree 1
    let tree1 = tree.visit_node(4).expect("failed to append children");
    tree.visit_node(100).expect("failed to append children");
    for _ in 0..100 {
        tree.visit_node(0).expect("failed to close children");
    }

    tree.visit_node(200).expect("failed to append children");
    for _ in 0..200 {
        tree.visit_node(0).expect("failed to close children");
    }

    tree.visit_node(400).expect("failed to append children");
    for _ in 0..400 {
        tree.visit_node(0).expect("failed to close children");
    }

    tree.visit_node(0).expect("failed to append children");

    // subtree 2
    let tree2 = tree.visit_node(1).expect("failed to append children");
    tree.visit_node(1).expect("failed to append children");
    tree.visit_node(1).expect("failed to append children");
    tree.visit_node(1).expect("failed to append children");
    tree.visit_node(500).expect("failed to append children");
    tree.visit_remaining_nodes();

    let tree = tree.build().expect("failed to build tree");

    assert_eq!(tree.subtree_size(root), 1211);
    assert_eq!(tree.subtree_size(tree1), 705);
    assert_eq!(tree.subtree_size(tree2), 505);
}
