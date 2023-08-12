use crate::trees::dfuds::{UDSTree, MIN_MAX_BLOCK_SIZE};
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
