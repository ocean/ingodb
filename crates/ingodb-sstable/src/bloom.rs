/// Simple bloom filter for SSTable key lookups.
///
/// Uses k=7 hash functions for ~1% false positive rate at 10 bits per key.
/// Accepts variable-length key bytes — hashes them via BLAKE3 truncation
/// to derive the bloom filter bit positions.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<u8>,
    num_bits: usize,
    num_hashes: u32,
}

const DEFAULT_BITS_PER_KEY: usize = 10;
const DEFAULT_NUM_HASHES: u32 = 7;

impl BloomFilter {
    /// Create a bloom filter sized for the expected number of keys.
    pub fn new(num_keys: usize) -> Self {
        let num_bits = (num_keys * DEFAULT_BITS_PER_KEY).max(64);
        let num_bytes = num_bits.div_ceil(8);
        BloomFilter {
            bits: vec![0u8; num_bytes],
            num_bits,
            num_hashes: DEFAULT_NUM_HASHES,
        }
    }

    /// Create from raw bytes (for deserialization).
    pub fn from_bytes(data: &[u8], num_bits: usize, num_hashes: u32) -> Self {
        BloomFilter {
            bits: data.to_vec(),
            num_bits,
            num_hashes,
        }
    }

    /// Derive two 64-bit hash values from arbitrary key bytes.
    /// Uses BLAKE3 to hash variable-length keys down to 16 bytes.
    fn hash_key(key: &[u8]) -> (u64, u64) {
        let h = blake3::hash(key);
        let bytes = h.as_bytes();
        let h1 = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let h2 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        (h1, h2)
    }

    /// Add a key to the filter.
    pub fn insert(&mut self, key: &[u8]) {
        let (h1, h2) = Self::hash_key(key);

        for i in 0..self.num_hashes {
            let bit_pos = (h1.wrapping_add(h2.wrapping_mul(i as u64))) as usize % self.num_bits;
            self.bits[bit_pos / 8] |= 1 << (bit_pos % 8);
        }
    }

    /// Check if a key might be in the filter.
    /// Returns false only if the key is definitely not present.
    pub fn may_contain(&self, key: &[u8]) -> bool {
        let (h1, h2) = Self::hash_key(key);

        for i in 0..self.num_hashes {
            let bit_pos = (h1.wrapping_add(h2.wrapping_mul(i as u64))) as usize % self.num_bits;
            if self.bits[bit_pos / 8] & (1 << (bit_pos % 8)) == 0 {
                return false;
            }
        }
        true
    }

    /// Raw bytes of the filter (for serialization).
    pub fn as_bytes(&self) -> &[u8] {
        &self.bits
    }

    /// Number of bits in the filter.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Number of hash functions.
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inserted_keys_found() {
        let mut bf = BloomFilter::new(100);
        let key1 = [0xAA; 16];
        let key2 = [0xBB; 16];

        bf.insert(&key1);
        bf.insert(&key2);

        assert!(bf.may_contain(&key1));
        assert!(bf.may_contain(&key2));
    }

    #[test]
    fn test_variable_length_keys() {
        let mut bf = BloomFilter::new(100);
        bf.insert(b"short");
        bf.insert(b"a much longer key value that exceeds 16 bytes");

        assert!(bf.may_contain(b"short"));
        assert!(bf.may_contain(b"a much longer key value that exceeds 16 bytes"));
        assert!(!bf.may_contain(b"not inserted"));
    }

    #[test]
    fn test_false_positive_rate() {
        let n = 1000;
        let mut bf = BloomFilter::new(n);

        // Insert n keys
        for i in 0..n {
            let key = format!("key-{i:06}");
            bf.insert(key.as_bytes());
        }

        // Check n keys that were NOT inserted
        let mut false_positives = 0;
        for i in n..2 * n {
            let key = format!("key-{i:06}");
            if bf.may_contain(key.as_bytes()) {
                false_positives += 1;
            }
        }

        // At 10 bits/key, 7 hashes, expect ~1% FP rate. Allow up to 5%.
        let fp_rate = false_positives as f64 / n as f64;
        assert!(
            fp_rate < 0.05,
            "false positive rate {fp_rate:.3} exceeds 5%"
        );
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut bf = BloomFilter::new(50);
        let key = [0xCC; 16];
        bf.insert(&key);

        let bf2 = BloomFilter::from_bytes(bf.as_bytes(), bf.num_bits(), bf.num_hashes());
        assert!(bf2.may_contain(&key));
    }
}
