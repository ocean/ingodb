use ingodb_blob::IBlob;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WalError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("CRC mismatch at offset {offset}: expected {expected:08x}, got {actual:08x}")]
    CrcMismatch {
        offset: u64,
        expected: u32,
        actual: u32,
    },

    #[error("blob decode error: {0}")]
    BlobDecode(#[from] ingodb_blob::BlobError),

    #[error("truncated record at offset {0}")]
    Truncated(u64),
}

/// Write-Ahead Log for IngoDB.
///
/// Record format:
/// ```text
/// [length: 4B LE]  — length of blob_bytes
/// [crc32:  4B LE]  — CRC32C of blob_bytes
/// [blob_bytes: length B]
/// ```
///
/// On recovery, corrupted or truncated tail records are skipped.
pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
    size: u64,
}

impl Wal {
    /// Open or create a WAL file at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WalError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let size = file.metadata()?.len();
        Ok(Wal {
            path,
            writer: BufWriter::new(file),
            size,
        })
    }

    /// Append multiple blobs to the WAL in one batch. Single flush at the end.
    pub fn append_batch(&mut self, blobs: &mut [IBlob]) -> Result<(), WalError> {
        for blob in blobs.iter_mut() {
            let blob_bytes = blob.encode();
            let length = blob_bytes.len() as u32;
            let crc = crc32fast::hash(&blob_bytes);
            self.writer.write_all(&length.to_le_bytes())?;
            self.writer.write_all(&crc.to_le_bytes())?;
            self.writer.write_all(&blob_bytes)?;
            self.size += 4 + 4 + blob_bytes.len() as u64;
        }
        self.writer.flush()?;
        Ok(())
    }

    /// Append a blob to the WAL. Returns the byte offset of the record.
    pub fn append(&mut self, blob: &mut IBlob) -> Result<u64, WalError> {
        let offset = self.size;
        let blob_bytes = blob.encode();
        let length = blob_bytes.len() as u32;
        let crc = crc32fast::hash(&blob_bytes);

        self.writer.write_all(&length.to_le_bytes())?;
        self.writer.write_all(&crc.to_le_bytes())?;
        self.writer.write_all(&blob_bytes)?;
        self.writer.flush()?;

        self.size += 4 + 4 + blob_bytes.len() as u64;
        Ok(offset)
    }

    /// Sync the WAL to disk (fsync).
    pub fn sync(&mut self) -> Result<(), WalError> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Current size of the WAL in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Recover all valid records from the WAL file.
    /// Stops at the first corrupted or truncated record.
    pub fn recover(path: impl AsRef<Path>) -> Result<Vec<IBlob>, WalError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);
        let mut blobs = Vec::new();
        let mut offset = 0u64;

        loop {
            if offset + 8 > file_len {
                break; // not enough room for header
            }

            // Read length
            let mut len_buf = [0u8; 4];
            if reader.read_exact(&mut len_buf).is_err() {
                break;
            }
            let length = u32::from_le_bytes(len_buf) as usize;

            // Read CRC
            let mut crc_buf = [0u8; 4];
            if reader.read_exact(&mut crc_buf).is_err() {
                break;
            }
            let stored_crc = u32::from_le_bytes(crc_buf);

            // Read blob bytes
            if offset + 8 + length as u64 > file_len {
                break; // truncated record
            }
            let mut blob_bytes = vec![0u8; length];
            if reader.read_exact(&mut blob_bytes).is_err() {
                break;
            }

            // Verify CRC
            let computed_crc = crc32fast::hash(&blob_bytes);
            if computed_crc != stored_crc {
                // Corrupted record — stop recovery here
                break;
            }

            match IBlob::decode(&blob_bytes) {
                Ok(blob) => blobs.push(blob),
                Err(_) => break, // corrupted blob data
            }

            offset += 8 + length as u64;
        }

        Ok(blobs)
    }

    /// Path to the WAL file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingodb_blob::{DocumentId, Value};

    fn make_blob(name: &str) -> IBlob {
        IBlob::from_pairs(vec![
            ("name", Value::String(name.into())),
            ("value", Value::U64(42)),
        ])
    }

    #[test]
    fn test_append_and_recover() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let mut blob1 = make_blob("first");
        let mut blob2 = make_blob("second");

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(&mut blob1).unwrap();
            wal.append(&mut blob2).unwrap();
            wal.sync().unwrap();
        }

        let recovered = Wal::recover(&wal_path).unwrap();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].id(), blob1.id());
        assert_eq!(recovered[0].hash(), blob1.hash());
        assert_eq!(recovered[1].id(), blob2.id());
        assert_eq!(recovered[1].hash(), blob2.hash());
    }

    #[test]
    fn test_recover_empty() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("empty.wal");
        let recovered = Wal::recover(&wal_path).unwrap();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_recover_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("trunc.wal");

        let mut blob1 = make_blob("good");

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(&mut blob1).unwrap();
            wal.sync().unwrap();
        }

        // Append garbage to simulate partial write
        {
            let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
            f.write_all(&[0xFF; 20]).unwrap();
        }

        let recovered = Wal::recover(&wal_path).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].hash(), blob1.hash());
    }

    #[test]
    fn test_recover_corrupted_crc() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("corrupt.wal");

        let mut blob1 = make_blob("good");
        let mut blob2 = make_blob("bad");

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(&mut blob1).unwrap();
            wal.append(&mut blob2).unwrap();
            wal.sync().unwrap();
        }

        // Corrupt the CRC of the second record
        {
            let mut data = std::fs::read(&wal_path).unwrap();
            let first_blob_bytes = blob1.encode();
            let second_record_start = 8 + first_blob_bytes.len();
            // CRC is at offset +4 within the record
            data[second_record_start + 4] ^= 0xFF;
            std::fs::write(&wal_path, data).unwrap();
        }

        let recovered = Wal::recover(&wal_path).unwrap();
        assert_eq!(recovered.len(), 1); // only first survives
        assert_eq!(recovered[0].hash(), blob1.hash());
    }

    #[test]
    fn test_tombstone_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("tomb.wal");

        let id = DocumentId::new();
        let mut tombstone = IBlob::tombstone(id);
        tombstone.set_version(DocumentId::new());

        {
            let mut wal = Wal::open(&wal_path).unwrap();
            wal.append(&mut tombstone).unwrap();
            wal.sync().unwrap();
        }

        let recovered = Wal::recover(&wal_path).unwrap();
        assert_eq!(recovered.len(), 1);
        assert!(recovered[0].is_deleted());
        assert_eq!(recovered[0].id(), &id);
    }
}
