use crate::{BitVec, RsVec};
use std::mem;

/// Encode a sequence of `n` `k`-bit words in a wavelet matrix.
/// The wavelet matrix allows for rank and select queries for `k`-bit symbols on the encoded sequence.
#[derive(Clone, Debug)]
pub struct WaveletMatrix {
    data: Box<[RsVec]>,
    bits_per_element: u16,
}

impl WaveletMatrix {
    /// Create a new wavelet matrix from a sequence of `n` `k`-bit words.
    ///
    /// # Parameters
    /// - `bit_vec`: The sequence of `n` `k`-bit words to encode. The `i`-th word begins in the
    ///   `bits_per_element * i`-th bit of the bit vector. Words are stored from least significant
    ///    bit to most significant bit.
    /// - `bits_per_element`: The number of bits in each word. Cannot exceed 1 << 16.
    ///
    /// # Panics
    /// Panics if the number of bits in the bit vector is not a multiple of the number of bits per element.
    #[must_use]
    pub fn from_bit_vec(bit_vec: &BitVec, bits_per_element: u16) -> Self {
        assert_eq!(bit_vec.len() % bits_per_element as usize, 0, "The number of bits in the bit vector must be a multiple of the number of bits per element.");
        let element_len = bits_per_element as usize;
        let num_elements = bit_vec.len() / element_len;

        let mut data = vec![BitVec::from_zeros(num_elements); element_len];

        // insert the first bit of each word into the first bit vector
        // for each following level, insert the next bit of each word into the next bit vector
        // sorted stably by the previous bit vector
        let mut permutation = (0..num_elements).collect::<Vec<_>>();
        let mut next_permutation = vec![0; num_elements];

        for level in 0..element_len {
            let mut total_zeros = 0;
            for i in 0..num_elements {
                if bit_vec.get_unchecked(permutation[i] * element_len + element_len - level - 1)
                    == 0
                {
                    total_zeros += 1;
                } else {
                    data[level].set(i, 1).unwrap();
                }
            }

            // scan through the generated bit array and move the elements to the correct position
            // for the next permutation
            if level < element_len - 1 {
                let mut zero_boundary = 0;
                let mut one_boundary = total_zeros;
                for i in 0..num_elements {
                    if data[level].get_unchecked(i) == 0 {
                        next_permutation[zero_boundary] = permutation[i];
                        zero_boundary += 1;
                    } else {
                        next_permutation[one_boundary] = permutation[i];
                        one_boundary += 1;
                    }
                }

                mem::swap(&mut permutation, &mut next_permutation);
            }
        }

        Self {
            data: data.into_iter().map(BitVec::into).collect(),
            bits_per_element,
        }
    }

    /// Generic function to read a value from the wavelet matrix and consume it with a closure.
    #[inline(always)]
    fn reconstruct_value_unchecked<F: FnMut(u64)>(&self, mut i: usize, mut target_func: F) {
        for level in 0..self.bits_per_element as usize {
            let bit = self.data[level].get_unchecked(i);
            target_func(bit);
            if bit == 0 {
                i = self.data[level].rank0(i);
            } else {
                i = self.data[level].rank0 + self.data[level].rank1(i);
            }
        }
    }

    /// Get the `i`-th element of the encoded sequence in a `k`-bit word.
    /// The `k`-bit word is returned as a `BitVec`.
    /// The first element of the bit vector is the least significant bit.
    #[must_use]
    pub fn get_value(&self, i: usize) -> Option<BitVec> {
        if self.data.is_empty() || i >= self.data[0].len() {
            None
        } else {
            Some(self.get_value_unchecked(i))
        }
    }

    /// Get the `i`-th element of the encoded sequence in a `k`-bit word.
    /// The `k`-bit word is returned as a `BitVec`.
    /// The first element of the bit vector is the least significant bit.
    /// This function does not perform bounds checking.
    /// Use [`get_value`] for a checked version.
    ///
    /// # Panics
    /// May panic if the number of bits per element exceeds 64. May instead return an empty bit vector.
    ///
    /// [`get_value`]: WaveletMatrix::get_value
    #[must_use]
    pub fn get_value_unchecked(&self, i: usize) -> BitVec {
        let mut value = BitVec::from_zeros(self.bits_per_element as usize);
        let mut idx = self.bits_per_element - 1;
        self.reconstruct_value_unchecked(i, |bit| {
            value.set_unchecked(idx as usize, bit);
            idx = idx.saturating_sub(1);
        });
        value
    }

    /// Get the `i`-th element of the encoded sequence as a `u64`.
    /// The `u64` is constructed from the `k`-bit word stored in the wavelet matrix.
    ///
    /// # Parameters
    /// - `i`: The index of the element to retrieve.
    ///
    /// # Panics
    /// Panics if the number of bits per element exceeds 64.
    #[must_use]
    pub fn get_u64(&self, i: usize) -> Option<u64> {
        if self.bits_per_element > 64 || self.data.is_empty() || i >= self.data[0].len() {
            None
        } else {
            Some(self.get_u64_unchecked(i))
        }
    }

    /// Get the `i`-th element of the encoded sequence as a `u64` numeral.
    /// The element is encoded in the lowest `k` bits of the numeral.
    /// If the number of bits per element exceeds 64, the value is truncated.
    /// This function does not perform bounds checking.
    /// Use [`get_u64`] for a checked version.
    ///
    /// # Panic
    /// May panic if the value of `i` is out of bounds. May instead return 0.
    ///
    /// [`get_u64`]: WaveletMatrix::get_u64
    #[must_use]
    pub fn get_u64_unchecked(&self, i: usize) -> u64 {
        let mut value = 0;
        self.reconstruct_value_unchecked(i, |bit| {
            value <<= 1;
            value |= bit;
        });
        value
    }

    /// Get the number of elements stored in the encoded sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        if self.data.is_empty() {
            0
        } else {
            self.data[0].len()
        }
    }

    /// Get the number of bytes allocated on the heap for the wavelet matrix.
    /// This does not include memory that is allocated but unused due to allocation policies of
    /// internal data structures.
    #[must_use]
    pub fn heap_size(&self) -> usize {
        self.data.iter().map(RsVec::heap_size).sum::<usize>()
    }
}

#[cfg(test)]
mod tests;
