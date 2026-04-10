use crate::bloom::BloomFilter;
use crate::error::SSTableError;
use crate::keys::KeyExtractor;
use crate::{DEFAULT_BLOCK_SIZE, SSTABLE_MAGIC, SSTABLE_VERSION};
use ingodb_blob::IBlob;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Writes an SSTable file from IBlobs, sorted by the provided KeyExtractor.
///
/// File layout:
/// ```text
/// [data block 0]  — each block: [crc32:4][compressed_data]
/// [data block 1]
/// ...
/// [data block N]
/// [bloom filter bytes]
/// [index entries]  — each: [key_len:2][key_bytes][offset:8][size:4]
/// [footer: 30B]
///   [index_offset: 8B]
///   [index_count: 4B]
///   [bloom_offset: 8B]
///   [bloom_size: 4B]
///   [magic: 4B "ISST"]
///   [version: 2B]
/// ```
///
/// Within each data block (after decompression):
/// ```text
/// [entry_count: 4B]
/// [key_len:2B][key_bytes][blob_len:4B][blob_bytes] ...repeated
/// ```
pub struct SSTableWriter {
    block_size: usize,
}

impl SSTableWriter {
    pub fn new() -> Self {
        SSTableWriter {
            block_size: DEFAULT_BLOCK_SIZE,
        }
    }

    pub fn with_block_size(block_size: usize) -> Self {
        SSTableWriter { block_size }
    }

    /// Write an SSTable. Entries are sorted by the KeyExtractor.
    pub fn write(
        &self,
        path: impl AsRef<Path>,
        blobs: &mut [IBlob],
        key_extractor: &dyn KeyExtractor,
    ) -> Result<(), SSTableError> {
        if blobs.is_empty() {
            return Err(SSTableError::Empty);
        }

        // Extract keys, sort by them
        let mut keyed: Vec<(Vec<u8>, usize)> = blobs
            .iter()
            .enumerate()
            .map(|(i, blob)| (key_extractor.extract_key(blob), i))
            .collect();
        keyed.sort_by(|a, b| a.0.cmp(&b.0));

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let mut block_index_entries: Vec<(Vec<u8>, u64, u32)> = Vec::new(); // (last_key, offset, size)
        let mut bloom = BloomFilter::new(blobs.len());

        // Build data blocks
        let mut current_block = Vec::new();
        let mut block_entry_count = 0u32;
        let mut file_offset = 0u64;
        let mut last_key_in_block = Vec::new();

        // Reserve space for entry count at start of block
        current_block.extend_from_slice(&0u32.to_le_bytes());

        for (key, idx) in &keyed {
            bloom.insert(key);
            // For MVCC keys (_id + _version = 32 bytes), also insert the _id prefix
            // so that point lookups by _id can use the bloom filter.
            if key.len() > 16 {
                bloom.insert(&key[..16]);
            }

            let blob_bytes = blobs[*idx].encode();
            let entry_size = 2 + key.len() + 4 + blob_bytes.len();

            // If adding this entry would exceed block size, flush current block
            if block_entry_count > 0 && current_block.len() + entry_size > self.block_size {
                let (offset, size) =
                    self.flush_block(&mut writer, &current_block, block_entry_count, file_offset)?;
                block_index_entries.push((last_key_in_block.clone(), offset, size));
                file_offset = offset + size as u64;

                // Reset block
                current_block.clear();
                current_block.extend_from_slice(&0u32.to_le_bytes());
                block_entry_count = 0;
            }

            // Write entry: [key_len:2][key_bytes][blob_len:4][blob_bytes]
            current_block.extend_from_slice(&(key.len() as u16).to_le_bytes());
            current_block.extend_from_slice(key);
            current_block.extend_from_slice(&(blob_bytes.len() as u32).to_le_bytes());
            current_block.extend_from_slice(&blob_bytes);
            last_key_in_block = key.clone();
            block_entry_count += 1;
        }

        // Flush final block
        if block_entry_count > 0 {
            let (offset, size) =
                self.flush_block(&mut writer, &current_block, block_entry_count, file_offset)?;
            block_index_entries.push((last_key_in_block, offset, size));
        }

        // Write bloom filter
        let bloom_offset = block_index_entries
            .last()
            .map(|(_, o, s)| *o + *s as u64)
            .unwrap_or(0);
        let bloom_bytes = bloom.as_bytes();
        let bloom_size = bloom_bytes.len() as u32 + 4 + 4;
        writer.write_all(&(bloom.num_bits() as u32).to_le_bytes())?;
        writer.write_all(&bloom.num_hashes().to_le_bytes())?;
        writer.write_all(bloom_bytes)?;

        // Write index entries: [key_len:2][key_bytes][offset:8][size:4]
        let index_offset = bloom_offset + bloom_size as u64;
        let index_count = block_index_entries.len() as u32;
        for (key, offset, size) in &block_index_entries {
            writer.write_all(&(key.len() as u16).to_le_bytes())?;
            writer.write_all(key)?;
            writer.write_all(&offset.to_le_bytes())?;
            writer.write_all(&size.to_le_bytes())?;
        }

        // Write footer
        writer.write_all(&index_offset.to_le_bytes())?;
        writer.write_all(&index_count.to_le_bytes())?;
        writer.write_all(&bloom_offset.to_le_bytes())?;
        writer.write_all(&bloom_size.to_le_bytes())?;
        writer.write_all(&SSTABLE_MAGIC)?;
        writer.write_all(&SSTABLE_VERSION.to_le_bytes())?;

        writer.flush()?;
        Ok(())
    }

    fn flush_block(
        &self,
        writer: &mut BufWriter<File>,
        block_data: &[u8],
        entry_count: u32,
        file_offset: u64,
    ) -> Result<(u64, u32), SSTableError> {
        // Patch entry count at the start of block data
        let mut data = block_data.to_vec();
        data[..4].copy_from_slice(&entry_count.to_le_bytes());

        // Compress with LZ4
        let compressed = lz4_flex::compress_prepend_size(&data);

        // CRC of compressed data
        let crc = crc32fast::hash(&compressed);

        // Write [crc:4][compressed_data]
        writer.write_all(&crc.to_le_bytes())?;
        writer.write_all(&compressed)?;

        let block_size = (4 + compressed.len()) as u32;
        Ok((file_offset, block_size))
    }
}

impl Default for SSTableWriter {
    fn default() -> Self {
        Self::new()
    }
}
