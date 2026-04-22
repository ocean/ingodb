use crate::bloom::BloomFilter;
use crate::error::SSTableError;
use crate::{FOOTER_SIZE, SSTABLE_MAGIC, SSTABLE_VERSION};
use ingodb_blob::{DocumentId, IBlob};
use std::fs;
use std::path::{Path, PathBuf};

/// Block index entry with variable-length key.
#[derive(Debug, Clone)]
struct BlockIndex {
    last_key: Vec<u8>,
    offset: u64,
    size: u32,
}

/// Reads and queries an SSTable file.
///
/// Keys are variable-length byte sequences. For primary SSTables the key
/// is the 16-byte DocumentId. For secondary indexes the key is the
/// comparable encoding of field values.
pub struct SSTableReader {
    path: PathBuf,
    data: Vec<u8>,
    block_indices: Vec<BlockIndex>,
    bloom: BloomFilter,
    min_key: Vec<u8>,
    max_key: Vec<u8>,
}

impl SSTableReader {
    /// Open an SSTable file and load its index and bloom filter into memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SSTableError> {
        let path = path.as_ref().to_path_buf();
        let data = fs::read(&path)?;

        if data.len() < FOOTER_SIZE {
            return Err(SSTableError::Corrupted("file too small for footer".into()));
        }

        // Parse footer (last FOOTER_SIZE bytes)
        let footer_start = data.len() - FOOTER_SIZE;
        let footer = &data[footer_start..];

        let index_offset = u64::from_le_bytes(footer[0..8].try_into().unwrap()) as usize;
        let index_count = u32::from_le_bytes(footer[8..12].try_into().unwrap()) as usize;
        let bloom_offset = u64::from_le_bytes(footer[12..20].try_into().unwrap()) as usize;
        let bloom_size = u32::from_le_bytes(footer[20..24].try_into().unwrap()) as usize;

        // Verify magic
        if footer[24..28] != SSTABLE_MAGIC {
            return Err(SSTableError::InvalidMagic);
        }
        let version = u16::from_le_bytes(footer[28..30].try_into().unwrap());
        if version != SSTABLE_VERSION {
            return Err(SSTableError::UnsupportedVersion(version));
        }

        // Validate offsets against footer start before indexing — footer values
        // come straight from disk and could be corrupt or adversarial.
        if index_offset > footer_start {
            return Err(SSTableError::Corrupted("index_offset past footer".into()));
        }
        let bloom_end = bloom_offset
            .checked_add(bloom_size)
            .ok_or_else(|| SSTableError::Corrupted("bloom bounds overflow".into()))?;
        if bloom_end > data.len() {
            return Err(SSTableError::Corrupted("bloom filter out of bounds".into()));
        }
        let bloom_data = &data[bloom_offset..bloom_end];
        if bloom_data.len() < 8 {
            return Err(SSTableError::Corrupted("bloom header truncated".into()));
        }
        let bloom_num_bits = u32::from_le_bytes(bloom_data[0..4].try_into().unwrap()) as usize;
        let bloom_num_hashes = u32::from_le_bytes(bloom_data[4..8].try_into().unwrap());
        let bloom = BloomFilter::from_bytes(&bloom_data[8..], bloom_num_bits, bloom_num_hashes);

        // Bound index_count against remaining space before allocating — a minimum
        // entry is 14 bytes (key_len:2 + offset:8 + size:4), so index_count larger
        // than that is definitely corrupt and must not drive Vec::with_capacity.
        const MIN_INDEX_ENTRY: usize = 14;
        let max_entries = footer_start.saturating_sub(index_offset) / MIN_INDEX_ENTRY;
        if index_count > max_entries {
            return Err(SSTableError::Corrupted(
                "index_count exceeds available space".into(),
            ));
        }

        // Parse block index — variable-length entries: [key_len:2][key_bytes][offset:8][size:4]
        let mut block_indices = Vec::with_capacity(index_count);
        let mut pos = index_offset;
        for _ in 0..index_count {
            let key_len_end = pos
                .checked_add(2)
                .ok_or_else(|| SSTableError::Corrupted("index offset overflow".into()))?;
            if key_len_end > footer_start {
                return Err(SSTableError::Corrupted("index entry truncated".into()));
            }
            let key_len = u16::from_le_bytes(data[pos..key_len_end].try_into().unwrap()) as usize;
            pos = key_len_end;
            let entry_end = pos
                .checked_add(key_len)
                .and_then(|p| p.checked_add(12))
                .ok_or_else(|| SSTableError::Corrupted("index offset overflow".into()))?;
            if entry_end > footer_start {
                return Err(SSTableError::Corrupted("index entry truncated".into()));
            }
            let last_key = data[pos..pos + key_len].to_vec();
            pos += key_len;
            let offset = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            block_indices.push(BlockIndex {
                last_key,
                offset,
                size,
            });
        }

        if block_indices.is_empty() {
            return Err(SSTableError::Corrupted("SSTable has no blocks".into()));
        }

        // max_key = last block's last key
        let max_key = block_indices.last().unwrap().last_key.clone();

        // min_key = first key of first block
        let min_key = Self::extract_first_key(&data, &block_indices[0])?;

        Ok(SSTableReader {
            path,
            data,
            block_indices,
            bloom,
            min_key,
            max_key,
        })
    }

    /// Extract the first key from a block without full parsing.
    fn extract_first_key(data: &[u8], block: &BlockIndex) -> Result<Vec<u8>, SSTableError> {
        let start = block.offset as usize;
        let end = start
            .checked_add(block.size as usize)
            .ok_or_else(|| SSTableError::Corrupted("block bounds overflow".into()))?;

        if end > data.len() {
            return Err(SSTableError::Corrupted("block out of bounds".into()));
        }

        let block_data = &data[start..end];
        if block_data.len() < 4 {
            return Err(SSTableError::Corrupted("block header truncated".into()));
        }
        let stored_crc = u32::from_le_bytes(block_data[0..4].try_into().unwrap());
        let compressed = &block_data[4..];
        let computed_crc = crc32fast::hash(compressed);
        if stored_crc != computed_crc {
            return Err(SSTableError::CrcMismatch {
                offset: block.offset,
            });
        }

        let decompressed = lz4_flex::decompress_size_prepended(compressed)
            .map_err(|e| SSTableError::Corrupted(format!("LZ4 decompress failed: {e}")))?;

        // First entry: [entry_count:4][key_len:2][key_bytes]...
        if decompressed.len() < 6 {
            return Err(SSTableError::Corrupted("block too small".into()));
        }
        let key_len = u16::from_le_bytes(decompressed[4..6].try_into().unwrap()) as usize;
        if decompressed.len() < 6 + key_len {
            return Err(SSTableError::Corrupted("first key truncated".into()));
        }
        Ok(decompressed[6..6 + key_len].to_vec())
    }

    /// Smallest key in this SSTable.
    pub fn min_key(&self) -> &[u8] {
        &self.min_key
    }

    /// Largest key in this SSTable.
    pub fn max_key(&self) -> &[u8] {
        &self.max_key
    }

    /// Convenience: min key as DocumentId (for primary SSTables).
    /// Works with both 16-byte (IdKey) and 32-byte (MvccKey) formats.
    pub fn min_id(&self) -> DocumentId {
        if self.min_key.len() >= 16 {
            let mut bytes = [0u8; 16];
            bytes.copy_from_slice(&self.min_key[..16]);
            DocumentId::from_bytes(bytes)
        } else {
            DocumentId::nil()
        }
    }

    /// Convenience: max key as DocumentId (for primary SSTables).
    pub fn max_id(&self) -> DocumentId {
        if self.max_key.len() >= 16 {
            let mut bytes = [0u8; 16];
            bytes.copy_from_slice(&self.max_key[..16]);
            DocumentId::from_bytes(bytes)
        } else {
            DocumentId::nil()
        }
    }

    /// File size in bytes.
    pub fn file_size(&self) -> u64 {
        self.data.len() as u64
    }

    /// Point lookup by key bytes.
    pub fn get(&self, key: &[u8]) -> Result<Option<IBlob>, SSTableError> {
        if !self.bloom.may_contain(key) {
            return Ok(None);
        }

        // Binary search block index
        let block_idx = match self
            .block_indices
            .binary_search_by(|idx| idx.last_key.as_slice().cmp(key))
        {
            Ok(i) => i,
            Err(i) => {
                if i >= self.block_indices.len() {
                    return Ok(None);
                }
                i
            }
        };

        let block = &self.block_indices[block_idx];
        let entries = self.read_block(block)?;

        match entries.binary_search_by(|(k, _)| k.as_slice().cmp(key)) {
            Ok(i) => Ok(Some(entries[i].1.clone())),
            Err(_) => Ok(None),
        }
    }

    /// Point lookup by DocumentId — returns the latest version.
    /// Works with both IdKeyExtractor (16-byte key) and MvccKeyExtractor (32-byte key).
    pub fn get_by_id(&self, id: &DocumentId) -> Result<Option<IBlob>, SSTableError> {
        // Try direct lookup first (IdKeyExtractor: 16-byte keys)
        if let Some(blob) = self.get(id.as_bytes())? {
            return Ok(Some(blob));
        }

        // Try MVCC key format: search for (_id + MAX_VERSION) and scan backward
        let mut mvcc_key = Vec::with_capacity(32);
        mvcc_key.extend_from_slice(id.as_bytes());
        mvcc_key.extend_from_slice(&[0xFF; 16]); // max version
        self.get_latest_by_prefix(id.as_bytes(), &mvcc_key)
    }

    /// Find the latest version of a document at a given snapshot.
    /// Returns the highest version <= snapshot for the given _id.
    pub fn get_by_id_at(
        &self,
        id: &DocumentId,
        snapshot: &DocumentId,
    ) -> Result<Option<IBlob>, SSTableError> {
        let mut mvcc_key = Vec::with_capacity(32);
        mvcc_key.extend_from_slice(id.as_bytes());
        mvcc_key.extend_from_slice(snapshot.as_bytes());
        self.get_latest_by_prefix(id.as_bytes(), &mvcc_key)
    }

    /// Find the latest entry whose key starts with `prefix` and is <= `max_key`.
    /// Used for MVCC lookups where the key is (_id, _version).
    fn get_latest_by_prefix(
        &self,
        prefix: &[u8],
        max_key: &[u8],
    ) -> Result<Option<IBlob>, SSTableError> {
        // Check bloom filter with the _id prefix (inserted at write time for MVCC keys)
        if !self.bloom.may_contain(prefix) {
            return Ok(None);
        }

        // Binary search for the block that could contain max_key
        let block_idx = match self
            .block_indices
            .binary_search_by(|idx| idx.last_key.as_slice().cmp(max_key))
        {
            Ok(i) => i,
            Err(i) => {
                if i >= self.block_indices.len() {
                    // max_key is beyond all blocks — check the last block
                    if self.block_indices.is_empty() {
                        return Ok(None);
                    }
                    self.block_indices.len() - 1
                } else {
                    i
                }
            }
        };

        // Search this block for entries with matching _id prefix and version <= max
        let id_prefix = &max_key[..16];
        let block = &self.block_indices[block_idx];
        let entries = self.read_block(block)?;

        // Scan backward to find latest version with matching _id prefix <= max_key
        let mut best: Option<IBlob> = None;
        for (k, blob) in entries.iter().rev() {
            if k.len() >= 16 && &k[..16] == id_prefix && k.as_slice() <= max_key {
                best = Some(blob.clone());
                break;
            }
        }

        // If not found in this block, check previous block (entry might be there)
        if best.is_none() && block_idx > 0 {
            let prev_block = &self.block_indices[block_idx - 1];
            if prev_block.last_key.len() >= 16 && &prev_block.last_key[..16] >= id_prefix {
                let entries = self.read_block(prev_block)?;
                for (k, blob) in entries.iter().rev() {
                    if k.len() >= 16 && &k[..16] == id_prefix && k.as_slice() <= max_key {
                        best = Some(blob.clone());
                        break;
                    }
                }
            }
        }

        Ok(best)
    }

    /// Iterate over all entries in key-sorted order.
    /// Returns (key_bytes, IBlob) pairs.
    pub fn iter(&self) -> Result<Vec<(Vec<u8>, IBlob)>, SSTableError> {
        let mut all_entries = Vec::new();
        for block in &self.block_indices {
            let entries = self.read_block(block)?;
            all_entries.extend(entries);
        }
        Ok(all_entries)
    }

    /// Range scan: return all entries with keys in [start_key, end_key].
    /// Both bounds are inclusive. Keys are compared lexicographically.
    pub fn range_scan(
        &self,
        start_key: &[u8],
        end_key: &[u8],
    ) -> Result<Vec<(Vec<u8>, IBlob)>, SSTableError> {
        // Find the first block that could contain start_key.
        // Binary search finds a block with last_key >= start_key, but entries
        // with the same key could span multiple blocks. Walk backward to find
        // the earliest block that could contain start_key.
        let mut first_block = match self
            .block_indices
            .binary_search_by(|idx| idx.last_key.as_slice().cmp(start_key))
        {
            Ok(i) => i,
            Err(i) => {
                if i >= self.block_indices.len() {
                    return Ok(Vec::new());
                }
                i
            }
        };
        // Walk backward: previous blocks might also contain entries >= start_key
        while first_block > 0 {
            if self.block_indices[first_block - 1].last_key.as_slice() >= start_key {
                first_block -= 1;
            } else {
                break;
            }
        }

        let mut results = Vec::new();
        for block_idx in first_block..self.block_indices.len() {
            let block = &self.block_indices[block_idx];
            let entries = self.read_block(block)?;

            let mut found_any_in_block = false;
            for (k, blob) in entries {
                if k.as_slice() < start_key {
                    continue;
                }
                if k.as_slice() > end_key {
                    return Ok(results); // past end of range — done
                }
                results.push((k, blob));
                found_any_in_block = true;
            }

            // If we read a full block past the start but found nothing in range,
            // and the block's last_key > end_key, no need to continue
            if !found_any_in_block && block.last_key.as_slice() > end_key {
                break;
            }
        }

        Ok(results)
    }

    /// Number of data blocks in this SSTable.
    pub fn block_count(&self) -> usize {
        self.block_indices.len()
    }

    /// Path to the SSTable file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read_block(&self, block: &BlockIndex) -> Result<Vec<(Vec<u8>, IBlob)>, SSTableError> {
        let start = block.offset as usize;
        let end = start
            .checked_add(block.size as usize)
            .ok_or_else(|| SSTableError::Corrupted("block bounds overflow".into()))?;

        if end > self.data.len() {
            return Err(SSTableError::Corrupted(format!(
                "block at offset {} extends past end of file",
                block.offset
            )));
        }

        let block_data = &self.data[start..end];

        // Verify CRC
        if block_data.len() < 4 {
            return Err(SSTableError::Corrupted("block header truncated".into()));
        }
        let stored_crc = u32::from_le_bytes(block_data[0..4].try_into().unwrap());
        let compressed = &block_data[4..];
        let computed_crc = crc32fast::hash(compressed);

        if stored_crc != computed_crc {
            return Err(SSTableError::CrcMismatch {
                offset: block.offset,
            });
        }

        // Decompress
        let decompressed = lz4_flex::decompress_size_prepended(compressed)
            .map_err(|e| SSTableError::Corrupted(format!("LZ4 decompress failed: {e}")))?;

        // Parse entries: [entry_count:4][key_len:2][key_bytes][blob_len:4][blob_bytes]...
        if decompressed.len() < 4 {
            return Err(SSTableError::Corrupted("block missing entry_count".into()));
        }
        let entry_count = u32::from_le_bytes(decompressed[0..4].try_into().unwrap()) as usize;
        // Bound entry_count before Vec::with_capacity — minimum per-entry cost
        // is 6 bytes (key_len:2 + blob_len:4), so a sane upper bound is
        // remaining bytes / 6. This prevents DoS via a crafted entry_count.
        const MIN_ENTRY_BYTES: usize = 6;
        let max_entries = decompressed.len().saturating_sub(4) / MIN_ENTRY_BYTES;
        if entry_count > max_entries {
            return Err(SSTableError::Corrupted(
                "entry_count exceeds block size".into(),
            ));
        }
        let mut entries = Vec::with_capacity(entry_count);
        let mut pos: usize = 4;

        for _ in 0..entry_count {
            let kl_end = pos
                .checked_add(2)
                .ok_or_else(|| SSTableError::Corrupted("block pos overflow".into()))?;
            if kl_end > decompressed.len() {
                return Err(SSTableError::Corrupted("truncated key_len in block".into()));
            }
            let key_len =
                u16::from_le_bytes(decompressed[pos..kl_end].try_into().unwrap()) as usize;
            pos = kl_end;

            let key_end = pos
                .checked_add(key_len)
                .ok_or_else(|| SSTableError::Corrupted("block pos overflow".into()))?;
            if key_end > decompressed.len() {
                return Err(SSTableError::Corrupted("truncated key in block".into()));
            }
            let key = decompressed[pos..key_end].to_vec();
            pos = key_end;

            let bl_end = pos
                .checked_add(4)
                .ok_or_else(|| SSTableError::Corrupted("block pos overflow".into()))?;
            if bl_end > decompressed.len() {
                return Err(SSTableError::Corrupted(
                    "truncated blob_len in block".into(),
                ));
            }
            let blob_len =
                u32::from_le_bytes(decompressed[pos..bl_end].try_into().unwrap()) as usize;
            pos = bl_end;

            let blob_end = pos
                .checked_add(blob_len)
                .ok_or_else(|| SSTableError::Corrupted("block pos overflow".into()))?;
            if blob_end > decompressed.len() {
                return Err(SSTableError::Corrupted("truncated blob in block".into()));
            }

            let blob = IBlob::decode(&decompressed[pos..blob_end])?;
            entries.push((key, blob));
            pos = blob_end;
        }

        Ok(entries)
    }
}
