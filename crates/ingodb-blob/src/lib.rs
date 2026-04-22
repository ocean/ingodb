mod error;
mod iblob;
mod value;

pub use error::BlobError;
pub use iblob::IBlob;
pub use value::Value;

/// 32-byte content hash (BLAKE3). Used for integrity verification and deduplication.
pub type ContentHash = [u8; 32];

/// Stable document identity — a UUIDv7 wrapped in a newtype for type safety.
///
/// UUIDv7 is timestamp-prefixed and lexicographically sortable, so documents
/// sort by creation time. Both client and server can generate these independently.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DocumentId([u8; 16]);

impl Default for DocumentId {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentId {
    /// Generate a new UUIDv7-based document ID (timestamp + random).
    pub fn new() -> Self {
        DocumentId(uuid::Uuid::now_v7().into_bytes())
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        DocumentId(bytes)
    }

    /// Access the raw bytes.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Maximum possible ID (all 0xFF bytes). Used as snapshot version meaning "latest".
    pub fn max() -> Self {
        DocumentId([0xFF; 16])
    }

    /// A nil (all-zeros) ID, used as a sentinel for unset `_version`.
    pub fn nil() -> Self {
        DocumentId([0; 16])
    }

    /// Check if this is the nil ID.
    pub fn is_nil(&self) -> bool {
        self.0 == [0; 16]
    }
}

impl std::fmt::Debug for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DocId({})", uuid::Uuid::from_bytes(self.0))
    }
}

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", uuid::Uuid::from_bytes(self.0))
    }
}

/// Magic bytes identifying an I-Blob on disk: "INGO"
pub const MAGIC: [u8; 4] = *b"INGO";

/// Current format version
pub const FORMAT_VERSION: u16 = 2;
