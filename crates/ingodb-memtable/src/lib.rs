use ingodb_blob::{DocumentId, IBlob};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// In-memory sorted table of IBlobs keyed by `(_id, _version)`.
///
/// Multiple versions of the same document coexist (MVCC).
/// Thread-safe via RwLock. Tracks approximate memory usage
/// and signals when it should be flushed to an SSTable.
pub struct MemTable {
    /// Entries keyed by (_id, _version) for multi-version support
    entries: RwLock<BTreeMap<(DocumentId, DocumentId), IBlob>>,
    /// Approximate bytes used (sum of encoded blob sizes)
    size_bytes: AtomicUsize,
    /// Flush threshold in bytes
    max_size: usize,
}

/// Iterator over memtable entries in (_id, _version) order.
pub struct MemTableIter {
    entries: Vec<(DocumentId, IBlob)>,
    pos: usize,
}

impl Iterator for MemTableIter {
    type Item = (DocumentId, IBlob);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.entries.len() {
            let item = self.entries[self.pos].clone();
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl MemTable {
    /// Create a new MemTable with the given flush threshold.
    pub fn new(max_size: usize) -> Self {
        MemTable {
            entries: RwLock::new(BTreeMap::new()),
            size_bytes: AtomicUsize::new(0),
            max_size,
        }
    }

    /// Default MemTable with 64 MB threshold.
    pub fn with_default_size() -> Self {
        Self::new(64 * 1024 * 1024)
    }

    /// Insert a blob. Multiple versions of the same _id coexist.
    /// Returns true if the memtable should be flushed.
    pub fn insert(&self, mut blob: IBlob) -> bool {
        let encoded_size = blob.encode().len();
        let key = (*blob.id(), *blob.version());

        let mut entries = self.entries.write();
        if let Some(mut old) = entries.insert(key, blob) {
            // Same _id + _version (unlikely but possible on replay) — adjust size
            let old_size = old.encode().len();
            self.size_bytes.fetch_sub(old_size, Ordering::Relaxed);
        }
        self.size_bytes.fetch_add(encoded_size, Ordering::Relaxed);

        self.size_bytes.load(Ordering::Relaxed) >= self.max_size
    }

    /// Look up the latest version of a document visible at `snapshot`.
    /// Returns the highest _version <= snapshot for the given _id.
    pub fn get(&self, id: &DocumentId, snapshot: &DocumentId) -> Option<IBlob> {
        let entries = self.entries.read();
        // Range scan: all entries with this _id, up to and including snapshot version
        let start = (*id, DocumentId::nil());
        let end = (*id, *snapshot);
        entries
            .range(start..=end)
            .next_back() // highest _version <= snapshot
            .map(|(_, blob)| blob.clone())
    }

    /// Look up the latest version of a document (any version).
    pub fn get_latest(&self, id: &DocumentId) -> Option<IBlob> {
        self.get(id, &DocumentId::max())
    }

    /// Check if a document ID exists in the memtable (any version).
    pub fn contains(&self, id: &DocumentId) -> bool {
        let entries = self.entries.read();
        let start = (*id, DocumentId::nil());
        let end = (*id, DocumentId::max());
        entries.range(start..=end).next().is_some()
    }

    /// Number of entries (including all versions).
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Whether the memtable is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Approximate size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size_bytes.load(Ordering::Relaxed)
    }

    /// Whether the memtable has reached its flush threshold.
    pub fn should_flush(&self) -> bool {
        self.size_bytes.load(Ordering::Relaxed) >= self.max_size
    }

    /// Drain all entries for flushing to an SSTable.
    /// Returns all versions, sorted by (_id, _version).
    pub fn drain(&self) -> Vec<IBlob> {
        let mut entries = self.entries.write();
        let drained: Vec<_> = entries.values().cloned().collect();
        entries.clear();
        self.size_bytes.store(0, Ordering::Relaxed);
        drained
    }

    /// Create an iterator over entries. Returns (_id, IBlob) pairs.
    /// Includes all versions.
    pub fn iter(&self) -> MemTableIter {
        let entries: Vec<_> = self
            .entries
            .read()
            .values()
            .map(|v| (*v.id(), v.clone()))
            .collect();
        MemTableIter { entries, pos: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingodb_blob::Value;

    fn make_blob(n: u64) -> IBlob {
        let mut blob = IBlob::from_pairs(vec![
            ("n", Value::U64(n)),
            ("data", Value::String(format!("item-{n}"))),
        ]);
        blob.set_version(DocumentId::new());
        blob
    }

    #[test]
    fn test_insert_and_get() {
        let mt = MemTable::new(1024 * 1024);
        let blob = make_blob(1);
        let id = *blob.id();

        mt.insert(blob.clone());
        let retrieved = mt.get_latest(&id).unwrap();
        assert_eq!(retrieved.id(), &id);
    }

    #[test]
    fn test_multi_version() {
        let mt = MemTable::new(1024 * 1024);
        let id = DocumentId::new();

        let mut blob1 = IBlob::with_id(id, [("x".into(), Value::U64(1))].into());
        blob1.set_version(DocumentId::new());
        let v1 = *blob1.version();

        let mut blob2 = IBlob::with_id(id, [("x".into(), Value::U64(2))].into());
        blob2.set_version(DocumentId::new());
        let v2 = *blob2.version();

        mt.insert(blob1);
        mt.insert(blob2);
        assert_eq!(mt.len(), 2, "both versions coexist");

        // Latest version
        let latest = mt.get_latest(&id).unwrap();
        assert_eq!(latest.get("x"), Some(&Value::U64(2)));

        // Snapshot at v1 — should see v1
        let old = mt.get(&id, &v1).unwrap();
        assert_eq!(old.get("x"), Some(&Value::U64(1)));

        // Snapshot at v2 — should see v2
        let current = mt.get(&id, &v2).unwrap();
        assert_eq!(current.get("x"), Some(&Value::U64(2)));
    }

    #[test]
    fn test_drain_all_versions() {
        let mt = MemTable::new(1024 * 1024);
        let id = DocumentId::new();

        let mut v1 = IBlob::with_id(id, [("x".into(), Value::U64(1))].into());
        v1.set_version(DocumentId::new());
        let mut v2 = IBlob::with_id(id, [("x".into(), Value::U64(2))].into());
        v2.set_version(DocumentId::new());

        mt.insert(v1);
        mt.insert(v2);

        let drained = mt.drain();
        assert_eq!(drained.len(), 2, "drain returns all versions");
        assert!(mt.is_empty());
    }

    #[test]
    fn test_drain_sorted() {
        let mt = MemTable::new(1024 * 1024);
        for i in 0..10 {
            mt.insert(make_blob(i));
        }

        let drained = mt.drain();
        assert_eq!(drained.len(), 10);
        assert!(mt.is_empty());
        assert_eq!(mt.size_bytes(), 0);
    }

    #[test]
    fn test_flush_threshold() {
        let mt = MemTable::new(1);
        let should = mt.insert(make_blob(1));
        assert!(should);
        assert!(mt.should_flush());
    }

    #[test]
    fn test_contains() {
        let mt = MemTable::new(1024 * 1024);
        let blob = make_blob(42);
        let id = *blob.id();

        assert!(!mt.contains(&id));
        mt.insert(blob);
        assert!(mt.contains(&id));
    }

    #[test]
    fn test_tombstone_visible_at_snapshot() {
        let mt = MemTable::new(1024 * 1024);
        let id = DocumentId::new();

        let mut blob = IBlob::with_id(id, [("x".into(), Value::U64(1))].into());
        blob.set_version(DocumentId::new());
        let v1 = *blob.version();
        mt.insert(blob);

        let mut tombstone = IBlob::tombstone(id);
        tombstone.set_version(DocumentId::new());
        mt.insert(tombstone);

        // Latest: tombstone
        let latest = mt.get_latest(&id).unwrap();
        assert!(latest.is_deleted());

        // Snapshot at v1: live document
        let old = mt.get(&id, &v1).unwrap();
        assert!(!old.is_deleted());
        assert_eq!(old.get("x"), Some(&Value::U64(1)));
    }
}
