use crate::bit_vec::fast_rs_vec::{BLOCK_SIZE, SUPER_BLOCK_SIZE};
use crate::bit_vec::fast_rs_vec::sealed::SealedDataAccess;
use crate::bit_vec::WORD_SIZE;
use crate::RsVec;

/// Defines the [`rank0`] and [`rank1`] methods for rank and select bitvector structs.
/// Also defines a range of other general methods.
/// This trait is sealed and cannot be implemented outside of this crate.
/// It exists to deduplicate code between the `RsVec` struct and its archived form, and can be used
/// to abstract over the different types.
///
/// [`rank0`]: RankSupport::rank0
/// [`rank1`]: RankSupport::rank1
pub trait RankSupport: SealedDataAccess {
    /// Return the total number of 0-bits in the bit-vector
    fn total_rank0(&self) -> usize;

    /// Return the total number of 1-bits in the bit-vector
    fn total_rank1(&self) -> usize;

    /// Return the 0-rank of the bit at the given position. The 0-rank is the number of
    /// 0-bits in the vector up to but excluding the bit at the given position. Calling this
    /// function with an index larger than the length of the bit-vector will report the total
    /// number of 0-bits in the bit-vector.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    #[must_use]
    fn rank0(&self, pos: usize) -> usize {
        self.rank(true, pos)
    }

    /// Return the 1-rank of the bit at the given position. The 1-rank is the number of
    /// 1-bits in the vector up to but excluding the bit at the given position. Calling this
    /// function with an index larger than the length of the bit-vector will report the total
    /// number of 1-bits in the bit-vector.
    ///
    /// # Parameters
    /// - `pos`: The position of the bit to return the rank of.
    #[must_use]
    fn rank1(&self, pos: usize) -> usize {
        self.rank(false, pos)
    }

    /// Return the rank of the bit at the given position. The rank is the number of
    /// zero or one bits in the vector up to but excluding the bit at the given position.
    /// Whether to use the 0-rank or 1-rank is determined by the `zero` parameter (true means zero).
    /// Calling this function with an index larger than the length of the bit-vector will report the
    /// total number of zero or one bits in the bit-vector.
    ///
    /// This function should be called through the [`rank0`] or [`rank1`] methods to benefit from
    /// optimization.
    // I measured 5-10% improvement with inlining. I don't know why it's not inlined by default, the
    // branch elimination profits alone should make it worth it.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn rank(&self, zero: bool, pos: usize) -> usize {
        #[allow(clippy::collapsible_else_if)]
        // readability and more obvious where dead branch elimination happens
        if zero {
            if pos >= self.len() {
                return self.total_rank0();
            }
        } else {
            if pos >= self.len() {
                return self.total_rank1();
            }
        }

        let index = pos / WORD_SIZE;
        let block_index = pos / BLOCK_SIZE;
        let super_block_index = pos / SUPER_BLOCK_SIZE;
        let mut rank = 0;

        // at first add the number of zeros/ones before the current super block
        rank += if zero {
            self.get_super_block_zeros(super_block_index)
        } else {
            (super_block_index * SUPER_BLOCK_SIZE) - self.get_super_block_zeros(super_block_index)
        };

        // then add the number of zeros/ones before the current block
        rank += if zero {
            self.get_block_zeros(block_index) as usize
        } else {
            ((block_index % (SUPER_BLOCK_SIZE / BLOCK_SIZE)) * BLOCK_SIZE)
                - self.get_block_zeros(block_index) as usize
        };

        // naive popcount of blocks
        for &i in self.get_data_range((block_index * BLOCK_SIZE) / WORD_SIZE..index) {
            rank += if zero {
                i.count_zeros() as usize
            } else {
                i.count_ones() as usize
            };
        }

        rank += if zero {
            (!self.get_data_word(index) & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as usize
        } else {
            (self.get_data_word(index) & ((1 << (pos % WORD_SIZE)) - 1)).count_ones() as usize
        };

        rank
    }

    /// Return the length of the vector, i.e. the number of bits it contains.
    #[must_use]
    fn len(&self) -> usize {
        self.bit_len()
    }

    /// Return whether the vector is empty.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the bit at the given position. The bit takes the least significant
    /// bit of the returned u64 word.
    /// If the position is larger than the length of the vector, `None` is returned.
    #[must_use]
    fn get(&self, pos: usize) -> Option<u64> {
        if pos >= self.len() {
            None
        } else {
            Some(self.get_unchecked(pos))
        }
    }

    /// Return the bit at the given position. The bit takes the least significant
    /// bit of the returned u64 word.
    ///
    /// # Panics
    /// This function may panic if `pos >= self.len()` (alternatively, it may return garbage).
    #[must_use]
    fn get_unchecked(&self, pos: usize) -> u64 {
        (self.get_data_word(pos / WORD_SIZE) >> (pos % WORD_SIZE)) & 1
    }

    /// Return multiple bits at the given position. The number of bits to return is given by `len`.
    /// At most 64 bits can be returned.
    /// If the position at the end of the query is larger than the length of the vector,
    /// None is returned (even if the query partially overlaps with the vector).
    /// If the length of the query is larger than 64, None is returned.
    #[must_use]
    fn get_bits(&self, pos: usize, len: usize) -> Option<u64> {
        if len > WORD_SIZE {
            return None;
        }
        if pos + len > self.bit_len() {
            None
        } else {
            Some(self.get_bits_unchecked(pos, len))
        }
    }

    /// Return multiple bits at the given position. The number of bits to return is given by `len`.
    /// At most 64 bits can be returned.
    ///
    /// This function is always inlined, because it gains a lot from loop optimization and
    /// can utilize the processor pre-fetcher better if it is.
    ///
    /// # Errors
    /// If the length of the query is larger than 64, unpredictable data will be returned.
    /// Use [`get_bits`] to properly handle this case with an `Option`.
    ///
    /// # Panics
    /// If the position or interval is larger than the length of the vector,
    /// the function will either return unpredictable data, or panic.
    ///
    /// [`get_bits`]: #method.get_bits
    #[must_use]
    #[allow(clippy::comparison_chain)] // rust-clippy #5354
    fn get_bits_unchecked(&self, pos: usize, len: usize) -> u64 {
        debug_assert!(len <= WORD_SIZE);
        let partial_word = self.get_data_word(pos / WORD_SIZE) >> (pos % WORD_SIZE);
        if pos % WORD_SIZE + len == WORD_SIZE {
            partial_word
        } else if pos % WORD_SIZE + len < WORD_SIZE {
            partial_word & ((1 << (len % WORD_SIZE)) - 1)
        } else {
            (partial_word
                | (self.get_data_word(pos / WORD_SIZE + 1) << (WORD_SIZE - pos % WORD_SIZE)))
                & ((1 << (len % WORD_SIZE)) - 1)
        }
    }

    /// Check if two `RsVec`s are equal. This compares limb by limb. This is usually faster than a
    /// [`sparse_equals`] call for small vectors.
    ///
    /// # Parameters
    /// - `other`: The other `RsVec` to compare to.
    ///
    /// # Returns
    /// `true` if the vectors' contents are equal, `false` otherwise.
    ///
    /// [`sparse_equals`]: RsVec::sparse_equals
    fn full_equals(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        if self.total_rank0() != other.total_rank0() || self.total_rank1() != other.total_rank1() {
            return false;
        }

        if self
            .get_data_range_to(..self.bit_len() / 64)
            .zip(other.get_data_range_to(..other.bit_len() / 64))
            .any(|(a, b)| a != b)
        {
            return false;
        }

        // if last incomplete block exists, test it without junk data
        if self.bit_len() % 64 > 0
            && self.get_data_word(self.bit_len() / 64) & ((1 << (self.bit_len() % 64)) - 1)
            != other.get_data_word(self.bit_len() / 64) & ((1 << (other.bit_len() % 64)) - 1)
        {
            return false;
        }

        true
    }
}

impl RankSupport for RsVec {
    fn total_rank0(&self) -> usize {
        self.rank0
    }

    fn total_rank1(&self) -> usize {
        self.rank1
    }
}