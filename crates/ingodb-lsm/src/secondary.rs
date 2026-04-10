use ingodb_blob::{DocumentId, IBlob, Value};
use ingodb_query::Filter;
use ingodb_sstable::{
    FieldKeyExtractor, KeyExtractor, SSTableReader, SSTableWriter, encode_comparable_value,
};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::LsmError;

/// Minimum number of results before sorting spills to disk as a partial index.
pub const SPILL_THRESHOLD: usize = 1000;

/// Number of compaction cycles before considering dropping an unused index.
pub const DROP_COMPACTION_CYCLES: u32 = 10;

/// Hours of wall-clock time before considering dropping an unused index.
pub const DROP_HOURS: u64 = 720; // 30 days

/// A secondary index: an SSTable sorted by field values instead of _id.
///
/// Each entry is a projected IBlob containing just the indexed fields.
/// The _id on each projected blob points back to the primary document.
/// Stale entries (from updated/deleted documents) are detected at read
/// time by verifying the indexed field values match the primary document.
///
/// Partial indexes have a `range` — the filter that produced them.
/// A full-range index (range=None) covers all documents.
pub struct SecondaryIndex {
    /// Fields this index covers (the sort key)
    pub fields: Vec<String>,
    /// The filter range this index covers. None = full range.
    pub range: Option<Filter>,
    /// The index SSTable (on-disk portion)
    pub reader: SSTableReader,
    /// Path to the index SSTable
    pub path: PathBuf,
    /// In-memory buffer of new index entries from put()
    buffer: Mutex<BTreeMap<Vec<u8>, IBlob>>,
    /// Last time this index was used for a query
    pub last_used: Mutex<Instant>,
    /// Number of compaction cycles since last use
    pub compactions_since_use: Mutex<u32>,
}

impl SecondaryIndex {
    /// Build a secondary index from primary SSTables.
    ///
    /// Reads all primary SSTables, deduplicates by _id (newest version wins),
    /// skips tombstones, projects to indexed fields, and writes an SSTable
    /// sorted by those field values.
    pub fn build(
        fields: &[String],
        primary_sstables: &[&SSTableReader],
        output_path: impl AsRef<Path>,
        block_size: usize,
    ) -> Result<Self, LsmError> {
        // Collect all blobs from primary SSTables
        let mut all: Vec<IBlob> = Vec::new();
        for sst in primary_sstables {
            all.extend(sst.iter()?.into_iter().map(|(_, blob)| blob));
        }

        // Dedup by _id, keep highest _version
        all.sort_by(|a, b| {
            a.id()
                .cmp(b.id())
                .then_with(|| b.version().cmp(a.version()))
        });
        all.dedup_by(|a, b| a.id() == b.id());

        // Skip tombstones and project to indexed fields
        let mut projected: Vec<IBlob> = all
            .into_iter()
            .filter(|blob| !blob.is_deleted())
            .map(|blob| blob.project(fields))
            .collect();

        if projected.is_empty() {
            return Err(LsmError::SSTable(ingodb_sstable::SSTableError::Empty));
        }

        // Write SSTable sorted by field values
        let extractor = FieldKeyExtractor::new(fields.to_vec());
        let output_path = output_path.as_ref().to_path_buf();
        SSTableWriter::with_block_size(block_size).write(
            &output_path,
            &mut projected,
            &extractor,
        )?;

        let reader = SSTableReader::open(&output_path)?;
        Ok(SecondaryIndex {
            fields: fields.to_vec(),
            range: None, // full range by default
            reader,
            path: output_path,
            buffer: Mutex::new(BTreeMap::new()),
            last_used: Mutex::new(Instant::now()),
            compactions_since_use: Mutex::new(0),
        })
    }

    /// Build a partial secondary index from pre-filtered, pre-sorted results.
    /// The `range` records which filter produced these results.
    pub fn build_partial(
        fields: &[String],
        range: Option<Filter>,
        sorted_blobs: &mut [IBlob],
        output_path: impl AsRef<Path>,
        block_size: usize,
    ) -> Result<Self, LsmError> {
        if sorted_blobs.is_empty() {
            return Err(LsmError::SSTable(ingodb_sstable::SSTableError::Empty));
        }

        // Project to indexed fields
        let mut projected: Vec<IBlob> = sorted_blobs
            .iter()
            .map(|blob| blob.project(fields))
            .collect();

        let extractor = FieldKeyExtractor::new(fields.to_vec());
        let output_path = output_path.as_ref().to_path_buf();
        SSTableWriter::with_block_size(block_size).write(
            &output_path,
            &mut projected,
            &extractor,
        )?;

        let reader = SSTableReader::open(&output_path)?;
        Ok(SecondaryIndex {
            fields: fields.to_vec(),
            range,
            reader,
            path: output_path,
            buffer: Mutex::new(BTreeMap::new()),
            last_used: Mutex::new(Instant::now()),
            compactions_since_use: Mutex::new(0),
        })
    }

    /// Open an existing secondary index from disk.
    pub fn open(
        fields: Vec<String>,
        range: Option<Filter>,
        path: impl AsRef<Path>,
    ) -> Result<Self, LsmError> {
        let path = path.as_ref().to_path_buf();
        let reader = SSTableReader::open(&path)?;
        Ok(SecondaryIndex {
            fields: fields.to_vec(),
            range,
            reader,
            path,
            buffer: Mutex::new(BTreeMap::new()),
            last_used: Mutex::new(Instant::now()),
            compactions_since_use: Mutex::new(0),
        })
    }

    /// Notify the index that a document was written/updated.
    /// If the document has the indexed fields, buffer a new index entry.
    pub fn notify_put(&self, blob: &IBlob) {
        let extractor = FieldKeyExtractor::new(self.fields.clone());
        let key = extractor.extract_key(blob);
        let projected = blob.project(&self.fields);
        self.buffer.lock().insert(key, projected);
    }

    /// Notify the index that a document was deleted.
    /// We don't remove from the buffer (stale entries handled on read).
    /// A tombstone entry is added to ensure the delete is visible.
    pub fn notify_delete(&self, id: &DocumentId) {
        // Tombstone projection: just the id, no fields
        let tomb = IBlob::tombstone(*id);
        // Use a key that will be found during scan
        // The stale check on read will handle it
        let extractor = FieldKeyExtractor::new(self.fields.clone());
        let key = extractor.extract_key(&tomb);
        self.buffer.lock().insert(key, tomb);
    }

    /// Iterate the index in sorted order, yielding (_id, projected IBlob) pairs.
    /// Merges on-disk SSTable entries with in-memory buffer.
    pub fn iter_sorted(&self) -> Result<Vec<(DocumentId, IBlob)>, LsmError> {
        // On-disk entries
        let mut entries: Vec<(Vec<u8>, IBlob)> = self.reader.iter()?;
        // Merge in-memory buffer
        let buffer = self.buffer.lock();
        for (key, blob) in buffer.iter() {
            entries.push((key.clone(), blob.clone()));
        }
        drop(buffer);

        // Sort by key (merge on-disk + buffer)
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(entries
            .into_iter()
            .map(|(_, blob)| (*blob.id(), blob))
            .collect())
    }

    /// Range scan the index: return entries where the indexed field value
    /// falls between start_value and end_value (inclusive).
    /// Uses binary search on the SSTable — O(log N + R) instead of O(N).
    pub fn range_scan(
        &self,
        start_value: Option<&Value>,
        end_value: Option<&Value>,
    ) -> Result<Vec<(DocumentId, IBlob)>, LsmError> {
        let start_key = start_value
            .map(encode_comparable_value)
            .unwrap_or_else(|| vec![0x00]); // min possible key
        let end_key = end_value
            .map(encode_comparable_value)
            .unwrap_or_else(|| vec![0xFF; 32]); // max possible key

        // SSTable range scan — binary search to start, scan to end
        let sst_entries = self.reader.range_scan(&start_key, &end_key)?;

        // Also include matching buffer entries
        let buffer = self.buffer.lock();
        let mut all: Vec<(Vec<u8>, IBlob)> = sst_entries;
        for (key, blob) in buffer.iter() {
            if key.as_slice() >= start_key.as_slice() && key.as_slice() <= end_key.as_slice() {
                all.push((key.clone(), blob.clone()));
            }
        }
        drop(buffer);

        // Sort merged results
        all.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(all
            .into_iter()
            .map(|(_, blob)| (*blob.id(), blob))
            .collect())
    }

    /// Check if this index covers the given sort fields AND the query filter.
    /// A full-range index (range=None) matches any filter.
    /// A partial index matches only if the query filter is exactly equal.
    pub fn matches_query(&self, sort_fields: &[String], query_filter: Option<&Filter>) -> bool {
        if self.fields != sort_fields {
            return false;
        }
        match (&self.range, query_filter) {
            (None, _) => true,                              // full range covers everything
            (Some(_), None) => false, // partial index can't serve unfiltered scan
            (Some(idx_range), Some(qf)) => idx_range == qf, // exact match
        }
    }

    /// Legacy: check sort fields only (for backward compat with existing tests).
    pub fn matches_sort(&self, sort_fields: &[String]) -> bool {
        self.fields == sort_fields
    }

    /// Mark this index as used (resets drop timer).
    pub fn mark_used(&self) {
        *self.last_used.lock() = Instant::now();
        *self.compactions_since_use.lock() = 0;
    }

    /// Mark a compaction cycle (for drop decisions).
    pub fn mark_compaction(&self) {
        *self.compactions_since_use.lock() += 1;
    }

    /// Should this index be dropped? True if unused for too long.
    pub fn should_drop(&self) -> bool {
        let cycles = *self.compactions_since_use.lock();
        let elapsed = self.last_used.lock().elapsed();
        cycles >= DROP_COMPACTION_CYCLES
            && elapsed >= std::time::Duration::from_secs(DROP_HOURS * 3600)
    }

    /// Total entry count (on-disk SSTable entries + in-memory buffer).
    pub fn entry_count(&self) -> u64 {
        let on_disk = self.reader.iter().map(|e| e.len() as u64).unwrap_or(0);
        let buffer = self.buffer.lock().len() as u64;
        on_disk + buffer
    }

    /// Compact the secondary index.
    ///
    /// Decision tree:
    /// 1. If `should_full_rebuild()`: rebuild from primary SSTables
    ///    - If index covers ≥50% of primary docs → extend to full range
    ///    - Otherwise → keep current partial range
    /// 2. Else: simple merge (flush buffer into index SSTable)
    ///
    /// Full rebuild when: `index_entries > 0.5 * primary_docs * ln(primary_docs)`
    pub fn compact(
        &mut self,
        primary_sstables: &[&SSTableReader],
        primary_doc_count: u64,
        block_size: usize,
    ) -> Result<(), LsmError> {
        self.mark_compaction();

        let index_entries = self.entry_count();

        if should_full_rebuild(index_entries, primary_doc_count) {
            // Decide range for rebuild
            let coverage = if primary_doc_count > 0 {
                index_entries as f64 / primary_doc_count as f64
            } else {
                1.0
            };

            if coverage >= 0.5 || self.range.is_none() {
                // Extend to full range — index already covers ≥50% or was full
                self.full_rebuild(primary_sstables, None, block_size)
            } else {
                // Keep current partial range
                self.full_rebuild(primary_sstables, self.range.clone(), block_size)
            }
        } else {
            self.merge_buffer(block_size)
        }
    }

    /// Full rebuild: re-read all primary SSTables, project, sort, write clean index.
    /// If `range` is Some, only include docs matching the filter.
    fn full_rebuild(
        &mut self,
        primary_sstables: &[&SSTableReader],
        range: Option<Filter>,
        block_size: usize,
    ) -> Result<(), LsmError> {
        let mut all: Vec<IBlob> = Vec::new();
        for sst in primary_sstables {
            all.extend(sst.iter()?.into_iter().map(|(_, blob)| blob));
        }

        all.sort_by(|a, b| {
            a.id()
                .cmp(b.id())
                .then_with(|| b.version().cmp(a.version()))
        });
        all.dedup_by(|a, b| a.id() == b.id());

        let mut projected: Vec<IBlob> = all
            .into_iter()
            .filter(|blob| !blob.is_deleted())
            .filter(|blob| {
                range
                    .as_ref()
                    .is_none_or(|f| f.matches(&|field| blob.get_field(field)))
            })
            .map(|blob| blob.project(&self.fields))
            .collect();

        if projected.is_empty() {
            return Ok(());
        }

        let extractor = FieldKeyExtractor::new(self.fields.clone());
        let tmp_path = self.path.with_extension("sst.tmp");
        SSTableWriter::with_block_size(block_size).write(&tmp_path, &mut projected, &extractor)?;

        std::fs::rename(&tmp_path, &self.path)?;
        self.reader = SSTableReader::open(&self.path)?;
        self.buffer.lock().clear();
        self.range = range;
        Ok(())
    }

    /// Simple merge: flush buffer entries into the existing index SSTable.
    fn merge_buffer(&mut self, block_size: usize) -> Result<(), LsmError> {
        let buffer = std::mem::take(&mut *self.buffer.lock());
        if buffer.is_empty() {
            return Ok(());
        }

        // Read existing entries
        let mut entries: Vec<(Vec<u8>, IBlob)> = self.reader.iter()?;

        // Add buffer entries
        for (key, blob) in buffer {
            entries.push((key, blob));
        }

        // Merge sort by key
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Extract just the blobs (writer will re-extract keys)
        let mut blobs: Vec<IBlob> = entries.into_iter().map(|(_, blob)| blob).collect();

        if blobs.is_empty() {
            return Ok(());
        }

        let extractor = FieldKeyExtractor::new(self.fields.clone());
        let tmp_path = self.path.with_extension("sst.tmp");
        SSTableWriter::with_block_size(block_size).write(&tmp_path, &mut blobs, &extractor)?;

        std::fs::rename(&tmp_path, &self.path)?;
        self.reader = SSTableReader::open(&self.path)?;
        Ok(())
    }
}

/// Determine if a full index rebuild is cheaper than a simple merge.
///
/// Full rebuild is O(N log N) where N = primary docs.
/// Simple merge is O(M) where M = index entries.
/// Exact crossover: M = N * ln(N).
/// We trigger at 0.5 * N * ln(N) to be proactive and avoid bloat.
pub fn should_full_rebuild(index_entries: u64, primary_docs: u64) -> bool {
    if primary_docs == 0 {
        return true;
    }
    let n = primary_docs as f64;
    index_entries as f64 > 0.5 * n * n.ln()
}

/// Default number of query executions before building an index reactively.
pub const DEFAULT_INDEX_THRESHOLD: u64 = 3;

#[cfg(test)]
mod tests {
    use super::*;
    use ingodb_blob::Value;
    use ingodb_sstable::IdKeyExtractor;

    fn make_primary_sst(dir: &Path, blobs: &mut [IBlob]) -> SSTableReader {
        let path = dir.join(format!("{}.sst", blobs[0].id()));
        SSTableWriter::new()
            .write(&path, blobs, &IdKeyExtractor)
            .unwrap();
        SSTableReader::open(&path).unwrap()
    }

    #[test]
    fn test_build_secondary_index() {
        let dir = tempfile::tempdir().unwrap();
        let mut blobs = vec![
            IBlob::from_pairs(vec![
                ("name", Value::String("Charlie".into())),
                ("age", Value::U64(30)),
            ]),
            IBlob::from_pairs(vec![
                ("name", Value::String("Alice".into())),
                ("age", Value::U64(25)),
            ]),
            IBlob::from_pairs(vec![
                ("name", Value::String("Bob".into())),
                ("age", Value::U64(35)),
            ]),
        ];
        // Stamp versions (normally done by engine)
        for blob in &mut blobs {
            blob.set_version(DocumentId::new());
        }

        let primary = make_primary_sst(dir.path(), &mut blobs);
        let idx_path = dir.path().join("idx_name.sst");

        let index = SecondaryIndex::build(&["name".into()], &[&primary], &idx_path, 4096).unwrap();

        // Iterate — should be sorted by name
        let sorted = index.iter_sorted().unwrap();
        let names: Vec<_> = sorted
            .iter()
            .filter_map(|(_, blob)| blob.get("name").cloned())
            .collect();
        assert_eq!(
            names,
            vec![
                Value::String("Alice".into()),
                Value::String("Bob".into()),
                Value::String("Charlie".into()),
            ]
        );
    }

    #[test]
    fn test_index_skips_tombstones() {
        let dir = tempfile::tempdir().unwrap();
        let mut live = IBlob::from_pairs(vec![("name", Value::String("Alice".into()))]);
        live.set_version(DocumentId::new());

        let mut tomb = IBlob::tombstone(DocumentId::new());
        tomb.set_version(DocumentId::new());

        let mut blobs = vec![live, tomb];
        let primary = make_primary_sst(dir.path(), &mut blobs);
        let idx_path = dir.path().join("idx_name.sst");

        let index = SecondaryIndex::build(&["name".into()], &[&primary], &idx_path, 4096).unwrap();

        let sorted = index.iter_sorted().unwrap();
        assert_eq!(sorted.len(), 1, "tombstone should be excluded from index");
    }

    #[test]
    fn test_index_matches_sort() {
        let dir = tempfile::tempdir().unwrap();
        let mut blobs = vec![IBlob::from_pairs(vec![("name", Value::String("A".into()))])];
        blobs[0].set_version(DocumentId::new());
        let primary = make_primary_sst(dir.path(), &mut blobs);

        let index = SecondaryIndex::build(
            &["name".into()],
            &[&primary],
            dir.path().join("idx.sst"),
            4096,
        )
        .unwrap();

        assert!(index.matches_sort(&["name".into()]));
        assert!(!index.matches_sort(&["age".into()]));
        assert!(!index.matches_sort(&["name".into(), "age".into()]));
    }

    #[test]
    fn test_index_open_existing() {
        let dir = tempfile::tempdir().unwrap();
        let mut blobs = vec![IBlob::from_pairs(vec![("x", Value::U64(1))])];
        blobs[0].set_version(DocumentId::new());
        let primary = make_primary_sst(dir.path(), &mut blobs);
        let idx_path = dir.path().join("idx.sst");

        SecondaryIndex::build(&["x".into()], &[&primary], &idx_path, 4096).unwrap();

        // Reopen
        let index = SecondaryIndex::open(vec!["x".into()], None, &idx_path).unwrap();
        assert_eq!(index.iter_sorted().unwrap().len(), 1);
    }

    #[test]
    fn test_should_full_rebuild_threshold() {
        // N=1000, ln(1000) ≈ 6.9, 0.5 * 1000 * 6.9 ≈ 3453
        assert!(!should_full_rebuild(3000, 1000)); // below threshold
        assert!(should_full_rebuild(4000, 1000)); // above threshold

        // N=0 → always rebuild
        assert!(should_full_rebuild(1, 0));

        // N=100, 0.5 * 100 * ln(100) ≈ 230
        assert!(!should_full_rebuild(200, 100));
        assert!(should_full_rebuild(250, 100));
    }

    #[test]
    fn test_merge_buffer_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let mut blobs = vec![
            IBlob::from_pairs(vec![("x", Value::U64(1))]),
            IBlob::from_pairs(vec![("x", Value::U64(3))]),
        ];
        for b in &mut blobs {
            b.set_version(DocumentId::new());
        }
        let primary = make_primary_sst(dir.path(), &mut blobs);
        let idx_path = dir.path().join("idx.sst");

        let mut index = SecondaryIndex::build(&["x".into()], &[&primary], &idx_path, 4096).unwrap();
        assert_eq!(index.entry_count(), 2);

        // Add buffer entry
        let mut new_blob = IBlob::from_pairs(vec![("x", Value::U64(2))]);
        new_blob.set_version(DocumentId::new());
        index.notify_put(&new_blob);
        assert_eq!(index.entry_count(), 3);

        // Merge compaction (3 < 0.5 * 3 * ln(3) ≈ 1.6, so NOT full rebuild — actually 3 > 1.6 so it IS rebuild)
        // With only 3 primary docs, threshold is 0.5*3*ln(3) ≈ 1.6, so 3 > 1.6 → full rebuild
        // Use a large primary_doc_count to force merge path
        index.compact(&[&primary], 10000, 4096).unwrap();

        // Buffer should be flushed
        assert_eq!(index.buffer.lock().len(), 0);
        // After merge, all entries should be in the SSTable
        assert!(index.entry_count() >= 3);
    }

    #[test]
    fn test_full_rebuild_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let mut blobs = vec![IBlob::from_pairs(vec![("x", Value::U64(1))])];
        blobs[0].set_version(DocumentId::new());
        let primary = make_primary_sst(dir.path(), &mut blobs);
        let idx_path = dir.path().join("idx.sst");

        let mut index = SecondaryIndex::build(&["x".into()], &[&primary], &idx_path, 4096).unwrap();

        // Add many buffer entries to bloat the index (simulating many updates)
        for i in 0..20u64 {
            let mut b = IBlob::from_pairs(vec![("x", Value::U64(i))]);
            b.set_version(DocumentId::new());
            index.notify_put(&b);
        }
        assert_eq!(index.entry_count(), 21); // 1 on-disk + 20 buffer

        // Full rebuild: 21 > 0.5 * 1 * ln(1) = 0 → rebuild
        index.compact(&[&primary], 1, 4096).unwrap();

        // After rebuild from primary, only 1 entry (the single primary doc)
        assert_eq!(index.entry_count(), 1);
        assert_eq!(index.buffer.lock().len(), 0);
    }

    #[test]
    fn test_compaction_extends_to_full_range() {
        // If a partial index covers ≥50% of primary docs, full rebuild extends to full range
        let dir = tempfile::tempdir().unwrap();
        let mut blobs: Vec<IBlob> = (0..10u64)
            .map(|i| {
                let mut b = IBlob::from_pairs(vec![("x", Value::U64(i))]);
                b.set_version(DocumentId::new());
                b
            })
            .collect();
        let primary = make_primary_sst(dir.path(), &mut blobs);
        let idx_path = dir.path().join("idx.sst");

        // Build partial index with 6 out of 10 entries (60% > 50%)
        let filter = Filter::Lt {
            field: "x".into(),
            value: Value::U64(6),
        };
        let mut partial_blobs: Vec<IBlob> = blobs
            .iter()
            .filter(|b| filter.matches(&|f| b.get_field(f)))
            .cloned()
            .collect();

        let mut index = SecondaryIndex::build_partial(
            &["x".into()],
            Some(filter),
            &mut partial_blobs,
            &idx_path,
            4096,
        )
        .unwrap();
        assert!(index.range.is_some(), "starts as partial");
        assert_eq!(index.entry_count(), 6);

        // Add buffer entries to trigger full rebuild
        for i in 0..20u64 {
            let mut b = IBlob::from_pairs(vec![("x", Value::U64(i + 100))]);
            b.set_version(DocumentId::new());
            index.notify_put(&b);
        }

        // Compact: 26 entries > 0.5 * 10 * ln(10) ≈ 11.5 → full rebuild
        // Coverage: 26/10 = 260% ≥ 50% → extend to full range
        index.compact(&[&primary], 10, 4096).unwrap();

        assert!(index.range.is_none(), "should extend to full range");
        assert_eq!(index.entry_count(), 10, "full rebuild from 10 primary docs");
    }

    #[test]
    fn test_compaction_keeps_partial_range() {
        // If partial index covers <50%, keep the partial range
        let dir = tempfile::tempdir().unwrap();
        let mut blobs: Vec<IBlob> = (0..100u64)
            .map(|i| {
                let mut b = IBlob::from_pairs(vec![("x", Value::U64(i))]);
                b.set_version(DocumentId::new());
                b
            })
            .collect();
        let primary = make_primary_sst(dir.path(), &mut blobs);
        let idx_path = dir.path().join("idx.sst");

        // Build partial index with only 3 out of 100 entries (3%)
        let filter = Filter::Lt {
            field: "x".into(),
            value: Value::U64(3),
        };
        let mut partial_blobs: Vec<IBlob> = blobs
            .iter()
            .filter(|b| filter.matches(&|f| b.get_field(f)))
            .cloned()
            .collect();

        let mut index = SecondaryIndex::build_partial(
            &["x".into()],
            Some(filter.clone()),
            &mut partial_blobs,
            &idx_path,
            4096,
        )
        .unwrap();

        // Add buffer entries to trigger rebuild
        for i in 0..50u64 {
            let mut b = IBlob::from_pairs(vec![("x", Value::U64(i + 1000))]);
            b.set_version(DocumentId::new());
            index.notify_put(&b);
        }

        // 53 entries > 0.5 * 100 * ln(100) ≈ 230 → NO, 53 < 230 → merge, not rebuild
        // So let's make a case where rebuild IS triggered but coverage < 50%:
        // Need M > 0.5 * N * ln(N) and M/N < 0.5
        // With N=100: threshold = 0.5 * 100 * 4.6 = 230. Need M > 230 but M/N < 0.5 → M < 50.
        // That's impossible (M > 230 and M < 50). So with large N, partial stays partial via merge.
        // This is actually correct behavior — large primary = simple merge is cheaper.
        index.compact(&[&primary], 100, 4096).unwrap();

        // Should still be partial (merge, not rebuild)
        assert!(index.range.is_some(), "should stay partial (merge path)");
    }

    #[test]
    fn test_should_drop_unused() {
        let dir = tempfile::tempdir().unwrap();
        let mut blobs = vec![IBlob::from_pairs(vec![("x", Value::U64(1))])];
        blobs[0].set_version(DocumentId::new());
        let primary = make_primary_sst(dir.path(), &mut blobs);

        let index =
            SecondaryIndex::build(&["x".into()], &[&primary], dir.path().join("idx.sst"), 4096)
                .unwrap();

        // Fresh index should not be dropped
        assert!(!index.should_drop());

        // After many compaction cycles but recently used — don't drop
        for _ in 0..DROP_COMPACTION_CYCLES + 1 {
            index.mark_compaction();
        }
        assert!(
            !index.should_drop(),
            "recently used — don't drop despite many compactions"
        );

        // Note: can't easily test the time-based drop in a unit test without
        // mocking time. The important thing is both conditions must be met.
    }
}
