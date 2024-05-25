//! A fast succinct bit vector implementation with rank and select queries. Rank computes in
//! constant-time, select on average in constant-time, with a logarithmic worst case.

use std::mem::size_of;
use std::ops::{Range, RangeTo};

use crate::bit_vec::fast_rs_vec::sealed::SealedDataAccess;
#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "avx512f",
    target_feature = "avx512bw",
))]
pub use bitset::*;
pub use iter::*;
pub use rank::*;
pub use select::*;

mod iter;
mod rank;
mod select;

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64",
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "avx512f",
    target_feature = "avx512bw",
))]
mod bitset;

use crate::util::impl_iterator;
use crate::BitVec;

use super::WORD_SIZE;

/// Size of a block in the bitvector.
const BLOCK_SIZE: usize = 512;

/// Size of a super block in the bitvector. Super-blocks exist to decrease the memory overhead
/// of block descriptors.
/// Increasing or decreasing the super block size has negligible effect on performance of rank
/// instruction. This means we want to make the super block size as large as possible, as long as
/// the zero-counter in normal blocks still fits in a reasonable amount of bits. However, this has
/// impact on the performance of select queries. The larger the super block size, the deeper will
/// a binary search be. We found 2^13 to be a good compromise between memory overhead and
/// performance.
const SUPER_BLOCK_SIZE: usize = 1 << 13;

/// Size of a select block. The select block is used to speed up select queries. The select block
/// contains the indices of every `SELECT_BLOCK_SIZE`'th 1-bit and 0-bit in the bitvector.
/// The smaller this block-size, the faster are select queries, but the more memory is used.
const SELECT_BLOCK_SIZE: usize = 1 << 13;

/// Meta-data for a block. The `zeros` field stores the number of zeros up to the block,
/// beginning from the last super-block boundary. This means the first block in a super-block
/// always stores the number zero, which serves as a sentinel value to avoid special-casing the
/// first block in a super-block (which would be a performance hit due branch prediction failures).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct BlockDescriptor {
    zeros: u16,
}

/// Meta-data for a super-block. The `zeros` field stores the number of zeros up to this super-block.
/// This allows the `BlockDescriptor` to store the number of zeros in a much smaller
/// space. The `zeros` field is the number of zeros up to the super-block.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct SuperBlockDescriptor {
    zeros: usize,
}

/// Meta-data for the select query. Each entry i in the select vector contains the indices to find
/// the i * `SELECT_BLOCK_SIZE`'th 0- and 1-bit in the bitvector. Those indices may be very far apart.
/// The indices do not point into the bit-vector, but into the select-block vector.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct SelectSuperBlockDescriptor {
    index_0: usize,
    index_1: usize,
}

mod sealed {
    use std::ops::{Range, RangeTo};

    pub trait SealedDataAccess {
        /// Return the amount of zeros stored in the super block at the given index.
        fn get_super_block_zeros(&self, idx: usize) -> usize;

        /// Return the length of the super block support vector.
        fn get_super_block_count(&self) -> usize;

        /// Return the amount of zeros stored in the block at the given index.
        fn get_block_zeros(&self, idx: usize) -> u16;

        /// Return the length of the block support vector.
        fn get_block_count(&self) -> usize;

        /// Return the data word at the given index.
        fn get_data_word(&self, idx: usize) -> u64;

        /// Return an iterator over the data words in the given range.
        fn get_data_range(&self, range: Range<usize>) -> impl Iterator<Item = &u64> + '_;

        /// Return an iterator over the data words in the given range.
        fn get_data_range_to(&self, range_from: RangeTo<usize>) -> impl Iterator<Item = &u64> + '_;

        /// Return the select block contents at the given index (index_0, index_1).
        fn get_select_block(&self, idx: usize) -> (usize, usize);

        /// Return the length of the vector, i.e. the number of bits it contains.
        fn bit_len(&self) -> usize;
    }
}

/// A bitvector that supports constant-time rank and select queries and is optimized for fast queries.
/// The bitvector is stored as a vector of `u64`s. The bit-vector stores meta-data for constant-time
/// rank and select queries, which takes sub-linear additional space. The space overhead is
/// 28 bits per 512 bits of user data (~5.47%).
///
/// # Example
/// ```rust
/// use vers_vecs::{BitVec, RsVec, RankSupport, SelectSupport};
///
/// let mut bit_vec = BitVec::new();
/// bit_vec.append_word(u64::MAX);
///
/// let rs_vec = RsVec::from_bit_vec(bit_vec);
/// assert_eq!(rs_vec.rank1(64), 64);
/// assert_eq!(rs_vec.select1(64), 64);
///```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RsVec {
    data: Vec<u64>,
    len: usize,
    blocks: Vec<BlockDescriptor>,
    super_blocks: Vec<SuperBlockDescriptor>,
    select_blocks: Vec<SelectSuperBlockDescriptor>,
    rank0: usize,
    rank1: usize,
}

impl RsVec {
    /// Build an [`RsVec`] from a [`BitVec`]. This will consume the [`BitVec`]. Since [`RsVec`]s are
    /// immutable, this is the only way to construct an [`RsVec`].
    ///
    /// # Example
    /// See the example for [`RsVec`].
    ///
    /// [`BitVec`]: ../struct.BitVec.html
    /// [`RsVec`]: struct.RsVec.html
    #[must_use]
    pub fn from_bit_vec(mut vec: BitVec) -> RsVec {
        // Construct the block descriptor meta data. Each block descriptor contains the number of
        // zeros in the super-block, up to but excluding the block.
        let mut blocks = Vec::with_capacity(vec.len() / BLOCK_SIZE + 1);
        let mut super_blocks = Vec::with_capacity(vec.len() / SUPER_BLOCK_SIZE + 1);
        let mut select_blocks = Vec::new();

        // sentinel value
        select_blocks.push(SelectSuperBlockDescriptor {
            index_0: 0,
            index_1: 0,
        });

        let mut total_zeros: usize = 0;
        let mut current_zeros: usize = 0;
        let mut last_zero_select_block: usize = 0;
        let mut last_one_select_block: usize = 0;

        for (idx, &word) in vec.data.iter().enumerate() {
            // if we moved past a block boundary, append the block information for the previous
            // block and reset the counter if we moved past a super-block boundary.
            if idx % (BLOCK_SIZE / WORD_SIZE) == 0 {
                if idx % (SUPER_BLOCK_SIZE / WORD_SIZE) == 0 {
                    total_zeros += current_zeros;
                    current_zeros = 0;
                    super_blocks.push(SuperBlockDescriptor { zeros: total_zeros });
                }

                // this cannot overflow because a super block isn't 2^16 bits long
                #[allow(clippy::cast_possible_truncation)]
                blocks.push(BlockDescriptor {
                    zeros: current_zeros as u16,
                });
            }

            // count the zeros in the current word and add them to the counter
            // the last word may contain padding zeros, which should not be counted,
            // but since we do not append the last block descriptor, this is not a problem
            let new_zeros = word.count_zeros() as usize;
            let all_zeros = total_zeros + current_zeros + new_zeros;
            if all_zeros / SELECT_BLOCK_SIZE > (total_zeros + current_zeros) / SELECT_BLOCK_SIZE {
                if all_zeros / SELECT_BLOCK_SIZE == select_blocks.len() {
                    select_blocks.push(SelectSuperBlockDescriptor {
                        index_0: super_blocks.len() - 1,
                        index_1: 0,
                    });
                } else {
                    select_blocks[all_zeros / SELECT_BLOCK_SIZE].index_0 = super_blocks.len() - 1;
                }

                last_zero_select_block += 1;
            }

            let total_bits = (idx + 1) * WORD_SIZE;
            let all_ones = total_bits - all_zeros;
            if all_ones / SELECT_BLOCK_SIZE
                > (idx * WORD_SIZE - total_zeros - current_zeros) / SELECT_BLOCK_SIZE
            {
                if all_ones / SELECT_BLOCK_SIZE == select_blocks.len() {
                    select_blocks.push(SelectSuperBlockDescriptor {
                        index_0: 0,
                        index_1: super_blocks.len() - 1,
                    });
                } else {
                    select_blocks[all_ones / SELECT_BLOCK_SIZE].index_1 = super_blocks.len() - 1;
                }

                last_one_select_block += 1;
            }

            current_zeros += new_zeros;
        }

        // insert dummy select blocks at the end that just report the same index like the last real
        // block, so the bound check for binary search doesn't overflow
        // this is technically the incorrect value, but since all valid queries will be smaller,
        // this will only tell select to stay in the current super block, which is correct.
        // we cannot use a real value here, because this would change the size of the super-block
        if last_zero_select_block == select_blocks.len() - 1 {
            select_blocks.push(SelectSuperBlockDescriptor {
                index_0: select_blocks[last_zero_select_block].index_0,
                index_1: 0,
            });
        } else {
            debug_assert!(select_blocks[last_zero_select_block + 1].index_0 == 0);
            select_blocks[last_zero_select_block + 1].index_0 =
                select_blocks[last_zero_select_block].index_0;
        }
        if last_one_select_block == select_blocks.len() - 1 {
            select_blocks.push(SelectSuperBlockDescriptor {
                index_0: 0,
                index_1: select_blocks[last_one_select_block].index_1,
            });
        } else {
            debug_assert!(select_blocks[last_one_select_block + 1].index_1 == 0);
            select_blocks[last_one_select_block + 1].index_1 =
                select_blocks[last_one_select_block].index_1;
        }

        // pad the internal vector to be block-aligned, so SIMD operations don't try to read
        // past the end of the vector. Note that this does not affect the content of the vector,
        // because those bits are not considered part of the vector.
        // Note further, that currently no SIMD implementation exists.
        while vec.data.len() % (BLOCK_SIZE / WORD_SIZE) != 0 {
            vec.data.push(0);
        }

        RsVec {
            data: vec.data,
            len: vec.len,
            blocks,
            super_blocks,
            select_blocks,
            // the last block may contain padding zeros, which should not be counted
            rank0: total_zeros + current_zeros - ((WORD_SIZE - (vec.len % WORD_SIZE)) % WORD_SIZE),
            rank1: vec.len
                - (total_zeros + current_zeros - ((WORD_SIZE - (vec.len % WORD_SIZE)) % WORD_SIZE)),
        }
    }

    /// Returns the number of bytes used on the heap for this vector. This does not include
    /// allocated space that is not used (e.g. by the allocation behavior of `Vec`).
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.data.len() * size_of::<u64>()
            + self.blocks.len() * size_of::<BlockDescriptor>()
            + self.super_blocks.len() * size_of::<SuperBlockDescriptor>()
            + self.select_blocks.len() * size_of::<SelectSuperBlockDescriptor>()
    }
}

impl SealedDataAccess for RsVec {
    fn get_super_block_zeros(&self, idx: usize) -> usize {
        self.super_blocks[idx].zeros
    }

    fn get_super_block_count(&self) -> usize {
        self.super_blocks.len()
    }

    fn get_block_zeros(&self, idx: usize) -> u16 {
        self.blocks[idx].zeros
    }

    fn get_block_count(&self) -> usize {
        self.blocks.len()
    }

    fn get_data_word(&self, idx: usize) -> u64 {
        self.data[idx]
    }

    fn get_data_range(&self, range: Range<usize>) -> impl Iterator<Item = &u64> + '_ {
        self.data[range].iter()
    }

    fn get_data_range_to(&self, range_from: RangeTo<usize>) -> impl Iterator<Item = &u64> + '_ {
        self.data[range_from].iter()
    }

    fn get_select_block(&self, idx: usize) -> (usize, usize) {
        (
            self.select_blocks[idx].index_0,
            self.select_blocks[idx].index_1,
        )
    }

    fn bit_len(&self) -> usize {
        self.len
    }
}

impl_iterator! { RsVec, RsVecIter, RsVecRefIter }

impl PartialEq for RsVec {
    /// Check if two `RsVec`s are equal. This method calls [`sparse_equals`] if the vector has more
    /// than 4'000'000 bits, and [`full_equals`] otherwise.
    ///
    /// This was determined with benchmarks on an `x86_64` machine,
    /// on which [`sparse_equals`] outperforms [`full_equals`] consistently above this threshold.
    ///
    /// # Parameters
    /// - `other`: The other `RsVec` to compare to.
    ///
    /// # Returns
    /// `true` if the vectors' contents are equal, `false` otherwise.
    ///
    /// [`sparse_equals`]: RsVec::sparse_equals
    /// [`full_equals`]: RsVec::full_equals
    fn eq(&self, other: &Self) -> bool {
        if self.len > 4000000 {
            if self.total_rank1() > self.total_rank0() {
                self.sparse_equals::<true>(other)
            } else {
                self.sparse_equals::<false>(other)
            }
        } else {
            self.full_equals(other)
        }
    }
}

#[cfg(test)]
mod tests;
