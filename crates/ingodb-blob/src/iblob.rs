use crate::{BlobError, ContentHash, DocumentId, FORMAT_VERSION, MAGIC, Value};
use std::collections::BTreeMap;

/// Index entry in the I-Blob header: maps a key hash to its location in the payload.
#[derive(Debug, Clone, Copy)]
struct IndexEntry {
    /// Hash of the field key (first 8 bytes of BLAKE3)
    key_hash: u64,
    /// Byte offset into the payload section
    offset: u32,
    /// Length of the encoded value in bytes
    length: u32,
}

const INDEX_ENTRY_SIZE: usize = 16; // 8 + 4 + 4
// header: magic(4) + format_version(2) + document_id(16) + doc_version(16) + flags(1) + content_hash(32) + index_count(4) = 75
const HEADER_SIZE: usize = 4 + 2 + 16 + 16 + 1 + 32 + 4;

/// IBlob flags (stored as 1 byte in the header)
const FLAG_DELETED: u8 = 0x01; // bit 0: tombstone
const FLAG_PROJECTION: u8 = 0x02; // bit 1: partial view of a document

/// An IngoDB document: a collection of named fields with stable identity.
///
/// Binary layout:
/// ```text
/// [magic: 4B "INGO"]
/// [format_version: 2B LE]
/// [document_id: 16B UUIDv7]
/// [doc_version: 16B UUIDv7]
/// [flags: 1B]
/// [content_hash: 32B BLAKE3]
/// [index_count: 4B LE]
/// [index_entries: index_count * 16B]
///   each: [key_hash: 8B LE][offset: 4B LE][length: 4B LE]
/// [payload: variable]
///   encoded Values concatenated in key-sorted order
/// ```
///
/// `_id` is the stable document identity (persists across updates).
/// `_version` is server-assigned at write time (advances on each update).
/// `flags` bitfield: bit 0 = deleted (tombstone), bit 1 = projection (partial view).
/// The content hash covers `_id` + `_version` + `flags` + fields for full integrity protection.
/// It is computed once at serialization time (`encode()`) or verified at deserialization (`decode()`).
#[derive(Debug, Clone)]
pub struct IBlob {
    /// Stable document identity (UUIDv7)
    id: DocumentId,
    /// Server-assigned write-order version (UUIDv7). Nil until server stamps it.
    version: DocumentId,
    /// Bitfield: FLAG_DELETED (0x01) = tombstone, FLAG_PROJECTION (0x02) = partial view
    flags: u8,
    /// BLAKE3 hash covering id + version + flags + fields.
    /// Zero until computed by encode() or decode().
    hash: ContentHash,
    /// Fields stored as sorted key-value pairs
    fields: BTreeMap<String, Value>,
    /// Cached serialized form
    encoded: Option<Vec<u8>>,
    /// Number of times compute_hash was called on this blob (for testing efficiency)
    hash_computations: u32,
}

impl IBlob {
    /// Create a new IBlob from key-value pairs.
    /// Auto-generates a UUIDv7 `_id`. The `_version` is left nil (server sets it at write time).
    /// Hash is computed later at encode() time.
    pub fn new(fields: BTreeMap<String, Value>) -> Self {
        IBlob {
            id: DocumentId::new(),
            version: DocumentId::nil(),
            flags: 0,
            hash: [0; 32],
            fields,
            encoded: None,
            hash_computations: 0,
        }
    }

    /// Create with a caller-supplied `_id`. The `_version` is left nil.
    pub fn with_id(id: DocumentId, fields: BTreeMap<String, Value>) -> Self {
        IBlob {
            id,
            version: DocumentId::nil(),
            flags: 0,
            hash: [0; 32],
            fields,
            encoded: None,
            hash_computations: 0,
        }
    }

    /// Create a tombstone for the given document ID.
    /// Hash is computed at encode() time.
    pub fn tombstone(id: DocumentId) -> Self {
        IBlob {
            id,
            version: DocumentId::nil(),
            flags: FLAG_DELETED,
            hash: [0; 32],
            fields: BTreeMap::new(),
            encoded: None,
            hash_computations: 0,
        }
    }

    /// Create a projected (partial) view of this document with only the specified fields.
    /// Same `_id` and `_version`, but with the projection flag set.
    pub fn project(&self, field_names: &[String]) -> IBlob {
        let fields: BTreeMap<String, Value> = field_names
            .iter()
            .filter_map(|name| self.fields.get(name).map(|v| (name.clone(), v.clone())))
            .collect();
        IBlob {
            id: self.id,
            version: self.version,
            flags: self.flags | FLAG_PROJECTION,
            hash: [0; 32],
            fields,
            encoded: None,
            hash_computations: 0,
        }
    }

    /// Create from a list of (key, value) tuples.
    pub fn from_pairs(pairs: Vec<(impl Into<String>, Value)>) -> Self {
        let fields: BTreeMap<String, Value> =
            pairs.into_iter().map(|(k, v)| (k.into(), v)).collect();
        Self::new(fields)
    }

    /// Stable document identity.
    pub fn id(&self) -> &DocumentId {
        &self.id
    }

    /// Server-assigned version (nil until written through the engine).
    pub fn version(&self) -> &DocumentId {
        &self.version
    }

    /// Set the version. Called by the LSM engine at write time.
    pub fn set_version(&mut self, v: DocumentId) {
        self.version = v;
        self.hash = [0; 32]; // invalidate — version is part of hash
        self.encoded = None;
    }

    /// Whether this is a tombstone (deleted document).
    pub fn is_deleted(&self) -> bool {
        self.flags & FLAG_DELETED != 0
    }

    /// Whether this is a projected (partial) view of a document.
    pub fn is_projection(&self) -> bool {
        self.flags & FLAG_PROJECTION != 0
    }

    /// Content hash covering id + version + flags + fields.
    /// Valid after encode() or decode(). Zero before first encode().
    pub fn hash(&self) -> &ContentHash {
        &self.hash
    }

    /// Access the fields.
    pub fn fields(&self) -> &BTreeMap<String, Value> {
        &self.fields
    }

    /// Get a field by name. Only looks in user fields, not metadata.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.fields.get(key)
    }

    /// Get a field by name, including virtual metadata fields `_id` and `_version`.
    /// Returns owned Value since metadata fields are constructed on the fly.
    pub fn get_field(&self, key: &str) -> Option<Value> {
        match key {
            "_id" => Some(Value::Uuid(self.id)),
            "_version" => Some(Value::Uuid(self.version)),
            _ => self.fields.get(key).cloned(),
        }
    }

    /// Number of fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Compute the key hash used in the index (first 8 bytes of BLAKE3 of the key).
    fn key_hash(key: &str) -> u64 {
        let h = blake3::hash(key.as_bytes());
        u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap())
    }

    /// Compute and store the BLAKE3 hash.
    fn compute_hash(&mut self) {
        self.hash_computations += 1;
        self.hash = *blake3::hash(&self.encode_hashable()).as_bytes();
    }

    /// Number of times the hash was computed on this blob (for testing).
    pub fn hash_computations(&self) -> u32 {
        self.hash_computations
    }

    /// Encode the hashable content: _id + _version + flags + fields.
    fn encode_hashable(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(self.id.as_bytes());
        buf.extend_from_slice(self.version.as_bytes());
        buf.push(self.flags);
        for (key, value) in &self.fields {
            buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            value.encode(&mut buf);
        }
        buf
    }

    /// Serialize this IBlob to its binary format.
    /// Computes the hash if not already computed.
    pub fn encode(&mut self) -> Vec<u8> {
        if let Some(ref cached) = self.encoded {
            return cached.clone();
        }

        // Compute hash once before writing
        if self.hash == [0; 32] {
            self.compute_hash();
        }

        // Encode each field
        let mut value_buffers: Vec<(String, u64, Vec<u8>)> = Vec::with_capacity(self.fields.len());
        for (key, value) in &self.fields {
            let kh = Self::key_hash(key);
            let mut vbuf = Vec::new();
            vbuf.extend_from_slice(&(key.len() as u32).to_le_bytes());
            vbuf.extend_from_slice(key.as_bytes());
            value.encode(&mut vbuf);
            value_buffers.push((key.clone(), kh, vbuf));
        }

        // Build the index entries and payload
        let index_count = value_buffers.len() as u32;
        let mut payload = Vec::new();
        let mut index_entries: Vec<IndexEntry> = Vec::with_capacity(value_buffers.len());

        for (_key, kh, vbuf) in &value_buffers {
            let offset = payload.len() as u32;
            let length = vbuf.len() as u32;
            index_entries.push(IndexEntry {
                key_hash: *kh,
                offset,
                length,
            });
            payload.extend_from_slice(vbuf);
        }

        // Sort index entries by key_hash for binary search during reads
        index_entries.sort_by_key(|e| e.key_hash);

        // Assemble the full blob
        let total_size = HEADER_SIZE + (index_entries.len() * INDEX_ENTRY_SIZE) + payload.len();
        let mut buf = Vec::with_capacity(total_size);

        // Header
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(self.id.as_bytes());
        buf.extend_from_slice(self.version.as_bytes());
        buf.push(self.flags);
        buf.extend_from_slice(&self.hash);
        buf.extend_from_slice(&index_count.to_le_bytes());

        // Index entries
        for entry in &index_entries {
            buf.extend_from_slice(&entry.key_hash.to_le_bytes());
            buf.extend_from_slice(&entry.offset.to_le_bytes());
            buf.extend_from_slice(&entry.length.to_le_bytes());
        }

        // Payload
        buf.extend_from_slice(&payload);

        self.encoded = Some(buf.clone());
        buf
    }

    /// Decode an IBlob from its binary format.
    /// Computes the hash to verify integrity.
    pub fn decode(buf: &[u8]) -> Result<Self, BlobError> {
        if buf.len() < HEADER_SIZE {
            return Err(BlobError::BufferTooShort {
                need: HEADER_SIZE,
                have: buf.len(),
            });
        }

        // Magic
        if buf[0..4] != MAGIC {
            return Err(BlobError::InvalidMagic);
        }

        // Format version
        let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(BlobError::UnsupportedVersion(version));
        }

        // Document ID (offset 6..22)
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&buf[6..22]);
        let id = DocumentId::from_bytes(id_bytes);

        // Document version (offset 22..38)
        let mut ver_bytes = [0u8; 16];
        ver_bytes.copy_from_slice(&buf[22..38]);
        let doc_version = DocumentId::from_bytes(ver_bytes);

        // flags (offset 38)
        let flags = buf[38];

        // Content hash (offset 39..71)
        let mut stored_hash = [0u8; 32];
        stored_hash.copy_from_slice(&buf[39..71]);

        // Index count (offset 71..75)
        let index_count = u32::from_le_bytes(buf[71..75].try_into().unwrap()) as usize;

        // Bound index_count against the remaining buffer before computing index_end.
        // A corrupt or malicious blob could otherwise overflow the multiplication
        // (on 32-bit targets) or pre-allocate a huge Vec below.
        let max_index_count = (buf.len() - HEADER_SIZE) / INDEX_ENTRY_SIZE;
        if index_count > max_index_count {
            return Err(BlobError::BufferTooShort {
                need: HEADER_SIZE.saturating_add(index_count.saturating_mul(INDEX_ENTRY_SIZE)),
                have: buf.len(),
            });
        }
        let index_end = HEADER_SIZE + index_count * INDEX_ENTRY_SIZE;

        // Read index entries
        let mut index_entries = Vec::with_capacity(index_count);
        for i in 0..index_count {
            let base = HEADER_SIZE + i * INDEX_ENTRY_SIZE;
            let key_hash = u64::from_le_bytes(buf[base..base + 8].try_into().unwrap());
            let offset = u32::from_le_bytes(buf[base + 8..base + 12].try_into().unwrap());
            let length = u32::from_le_bytes(buf[base + 12..base + 16].try_into().unwrap());
            index_entries.push(IndexEntry {
                key_hash,
                offset,
                length,
            });
        }

        // Payload starts after the index
        let payload = &buf[index_end..];

        // Decode fields from payload
        let mut fields = BTreeMap::new();
        for entry in &index_entries {
            let start = entry.offset as usize;
            let end = start + entry.length as usize;
            if end > payload.len() {
                return Err(BlobError::BufferTooShort {
                    need: index_end + end,
                    have: buf.len(),
                });
            }
            let field_data = &payload[start..end];

            // Decode key
            if field_data.len() < 4 {
                return Err(BlobError::BufferTooShort {
                    need: 4,
                    have: field_data.len(),
                });
            }
            let key_len = u32::from_le_bytes(field_data[0..4].try_into().unwrap()) as usize;
            if field_data.len() < 4 + key_len {
                return Err(BlobError::BufferTooShort {
                    need: 4 + key_len,
                    have: field_data.len(),
                });
            }
            let key = std::str::from_utf8(&field_data[4..4 + key_len])?.to_string();

            // Decode value
            let (value, _consumed) = Value::decode(field_data, 4 + key_len)?;
            fields.insert(key, value);
        }

        // Verify hash — compute once for integrity check
        let mut blob = IBlob {
            id,
            version: doc_version,
            flags,
            hash: [0; 32],
            fields,
            encoded: None,
            hash_computations: 0,
        };
        blob.compute_hash();
        if blob.hash != stored_hash {
            return Err(BlobError::HashMismatch {
                expected: hex(&blob.hash),
                actual: hex(&stored_hash),
            });
        }

        Ok(blob)
    }

    /// Zero-copy field extraction: read a single field from a serialized blob
    /// without decoding the entire document.
    pub fn extract_field(buf: &[u8], field_name: &str) -> Result<Value, BlobError> {
        if buf.len() < HEADER_SIZE {
            return Err(BlobError::BufferTooShort {
                need: HEADER_SIZE,
                have: buf.len(),
            });
        }

        let index_count = u32::from_le_bytes(buf[71..75].try_into().unwrap()) as usize;

        // Bound index_count against the remaining buffer so a corrupt header
        // can't trigger a panic in the slice operations below.
        let max_index_count = (buf.len() - HEADER_SIZE) / INDEX_ENTRY_SIZE;
        if index_count > max_index_count {
            return Err(BlobError::BufferTooShort {
                need: HEADER_SIZE.saturating_add(index_count.saturating_mul(INDEX_ENTRY_SIZE)),
                have: buf.len(),
            });
        }
        let index_end = HEADER_SIZE + index_count * INDEX_ENTRY_SIZE;

        let target_hash = Self::key_hash(field_name);

        // Binary search the index
        let mut lo = 0usize;
        let mut hi = index_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let base = HEADER_SIZE + mid * INDEX_ENTRY_SIZE;
            let kh = u64::from_le_bytes(buf[base..base + 8].try_into().unwrap());
            if kh < target_hash {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // Check for match (may need to scan duplicates if key_hash collides)
        while lo < index_count {
            let base = HEADER_SIZE + lo * INDEX_ENTRY_SIZE;
            let kh = u64::from_le_bytes(buf[base..base + 8].try_into().unwrap());
            if kh != target_hash {
                break;
            }
            let offset = u32::from_le_bytes(buf[base + 8..base + 12].try_into().unwrap()) as usize;
            let length = u32::from_le_bytes(buf[base + 12..base + 16].try_into().unwrap()) as usize;

            let payload_start = index_end
                .checked_add(offset)
                .ok_or(BlobError::BufferTooShort {
                    need: 0,
                    have: buf.len(),
                })?;
            let payload_end =
                payload_start
                    .checked_add(length)
                    .ok_or(BlobError::BufferTooShort {
                        need: 0,
                        have: buf.len(),
                    })?;
            if payload_end > buf.len() {
                return Err(BlobError::BufferTooShort {
                    need: payload_end,
                    have: buf.len(),
                });
            }
            let field_data = &buf[payload_start..payload_end];

            // Check actual key
            if field_data.len() < 4 {
                return Err(BlobError::BufferTooShort {
                    need: 4,
                    have: field_data.len(),
                });
            }
            let key_len = u32::from_le_bytes(field_data[0..4].try_into().unwrap()) as usize;
            let key_end = 4usize
                .checked_add(key_len)
                .ok_or(BlobError::BufferTooShort {
                    need: 0,
                    have: field_data.len(),
                })?;
            if key_end > field_data.len() {
                return Err(BlobError::BufferTooShort {
                    need: key_end,
                    have: field_data.len(),
                });
            }
            let key = std::str::from_utf8(&field_data[4..key_end])?;

            if key == field_name {
                let (value, _) = Value::decode(field_data, key_end)?;
                return Ok(value);
            }
            lo += 1;
        }

        Err(BlobError::FieldNotFound(target_hash))
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

impl PartialEq for IBlob {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for IBlob {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;

    fn sample_blob() -> IBlob {
        IBlob::from_pairs(vec![
            ("name", Value::String("Henrik".into())),
            ("age", Value::U64(42)),
            ("active", Value::Bool(true)),
        ])
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut blob = sample_blob();
        let encoded = blob.encode();
        let decoded = IBlob::decode(&encoded).unwrap();

        assert_eq!(blob.id(), decoded.id());
        assert_eq!(blob.hash(), decoded.hash());
        assert_eq!(blob.fields(), decoded.fields());
        assert!(!decoded.is_deleted());
    }

    #[test]
    fn test_hash_deterministic() {
        // Same _id + same fields = same hash (deterministic)
        let id = DocumentId::from_bytes([0x42; 16]);
        let mut blob1 = IBlob::with_id(id, [("x".into(), Value::I64(1))].into());
        let mut blob2 = IBlob::with_id(id, [("x".into(), Value::I64(1))].into());
        blob1.encode();
        blob2.encode();
        assert_eq!(blob1.hash(), blob2.hash());
    }

    #[test]
    fn test_different_content_different_hash() {
        let mut blob1 = IBlob::from_pairs(vec![("x", Value::I64(1))]);
        let mut blob2 = IBlob::from_pairs(vec![("x", Value::I64(2))]);
        blob1.encode();
        blob2.encode();
        assert_ne!(blob1.hash(), blob2.hash());
    }

    #[test]
    fn test_explicit_id() {
        let id = DocumentId::from_bytes([0x42; 16]);
        let fields: BTreeMap<String, Value> = [("k".into(), Value::U64(1))].into();
        let mut blob = IBlob::with_id(id, fields);
        assert_eq!(*blob.id(), id);

        // Roundtrip preserves the id
        let encoded = blob.encode();
        let decoded = IBlob::decode(&encoded).unwrap();
        assert_eq!(*decoded.id(), id);
    }

    #[test]
    fn test_version_roundtrip() {
        let mut blob = sample_blob();
        assert!(blob.version().is_nil(), "version starts nil");

        let ver = DocumentId::new();
        blob.set_version(ver);
        assert_eq!(*blob.version(), ver);

        let encoded = blob.encode();
        let decoded = IBlob::decode(&encoded).unwrap();
        assert_eq!(*decoded.version(), ver);
    }

    #[test]
    fn test_tombstone_roundtrip() {
        let id = DocumentId::new();
        let mut tomb = IBlob::tombstone(id);
        assert!(tomb.is_deleted());
        assert_eq!(tomb.field_count(), 0);

        // Stamp a version like the engine would
        tomb.set_version(DocumentId::new());

        let encoded = tomb.encode();
        assert_ne!(
            *tomb.hash(),
            [0u8; 32],
            "tombstone should have a real hash after encode"
        );

        let decoded = IBlob::decode(&encoded).unwrap();
        assert_eq!(decoded.id(), &id);
        assert!(decoded.is_deleted());
        assert_eq!(decoded.field_count(), 0);
        assert_eq!(decoded.version(), tomb.version());
        assert_eq!(decoded.hash(), tomb.hash());
    }

    #[test]
    fn test_projection() {
        let mut blob = sample_blob();
        blob.set_version(DocumentId::new());
        blob.encode(); // compute hash

        assert!(!blob.is_projection());

        let mut projected = blob.project(&["name".into(), "age".into()]);
        assert!(projected.is_projection());
        assert!(!projected.is_deleted());
        assert_eq!(projected.field_count(), 2);
        assert_eq!(projected.get("name"), Some(&Value::String("Henrik".into())));
        assert_eq!(projected.get("age"), Some(&Value::U64(42)));
        assert!(
            projected.get("active").is_none(),
            "unprojected field absent"
        );

        // Same _id and _version as original
        assert_eq!(projected.id(), blob.id());
        assert_eq!(projected.version(), blob.version());

        // Roundtrips correctly
        let encoded = projected.encode();
        let decoded = IBlob::decode(&encoded).unwrap();
        assert!(decoded.is_projection());
        assert_eq!(decoded.field_count(), 2);
        assert_eq!(decoded.id(), blob.id());
    }

    #[test]
    fn test_projection_missing_fields() {
        let blob = sample_blob();
        let projected = blob.project(&["nonexistent".into()]);
        assert_eq!(projected.field_count(), 0);
        assert!(projected.is_projection());
    }

    #[test]
    fn test_extract_field() {
        let mut blob = sample_blob();
        let encoded = blob.encode();

        let name = IBlob::extract_field(&encoded, "name").unwrap();
        assert_eq!(name, Value::String("Henrik".into()));

        let age = IBlob::extract_field(&encoded, "age").unwrap();
        assert_eq!(age, Value::U64(42));

        assert!(IBlob::extract_field(&encoded, "nonexistent").is_err());
    }

    #[test]
    fn test_nested_document_blob() {
        let mut blob = IBlob::from_pairs(vec![
            (
                "user",
                Value::Document(vec![
                    ("name".into(), Value::String("Henrik".into())),
                    ("email".into(), Value::String("henrik@example.com".into())),
                ]),
            ),
            (
                "refs",
                Value::Array(vec![
                    Value::Uuid(DocumentId::from_bytes([0xAA; 16])),
                    Value::Uuid(DocumentId::from_bytes([0xBB; 16])),
                ]),
            ),
        ]);

        let encoded = blob.encode();
        let decoded = IBlob::decode(&encoded).unwrap();
        assert_eq!(blob.hash(), decoded.hash());
        assert_eq!(blob.fields(), decoded.fields());
    }

    #[test]
    fn test_empty_blob() {
        let mut blob = IBlob::new(BTreeMap::new());
        let encoded = blob.encode();
        let decoded = IBlob::decode(&encoded).unwrap();
        assert_eq!(decoded.field_count(), 0);
        assert_eq!(blob.hash(), decoded.hash());
    }

    #[test]
    fn test_invalid_magic() {
        let mut encoded = sample_blob().encode();
        encoded[0] = b'X';
        assert!(matches!(
            IBlob::decode(&encoded),
            Err(BlobError::InvalidMagic)
        ));
    }

    #[test]
    fn test_hash_computed_once_per_encode() {
        let mut blob = IBlob::from_pairs(vec![("x", Value::U64(1))]);
        blob.set_version(DocumentId::new());
        assert_eq!(blob.hash_computations(), 0, "no hash at construction");

        blob.encode();
        assert_eq!(blob.hash_computations(), 1, "hash computed once by encode");

        blob.encode(); // cached
        assert_eq!(blob.hash_computations(), 1, "second encode uses cache");
    }

    #[test]
    fn test_hash_computed_once_for_put_path() {
        // Simulates engine put: construct → set_version → encode (WAL) → encode (memtable size)
        let mut blob = IBlob::from_pairs(vec![
            ("name", Value::String("test".into())),
            ("value", Value::U64(42)),
        ]);
        blob.set_version(DocumentId::new());
        blob.encode();
        blob.encode();
        assert_eq!(
            blob.hash_computations(),
            1,
            "put path computes hash exactly once"
        );
    }
}
