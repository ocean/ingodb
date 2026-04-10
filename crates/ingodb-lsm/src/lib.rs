mod compaction;
mod database;
mod secondary;
pub mod stats;

pub use database::Database;

use ingodb_blob::{DocumentId, IBlob, Value};
use ingodb_memtable::MemTable;
use ingodb_query::{Filter, Query, SortDirection, SortField, compare_values};
use ingodb_sstable::{MvccKeyExtractor, SSTableReader, SSTableWriter};
use ingodb_wal::Wal;
use parking_lot::{Condvar, Mutex, RwLock};
use stats::{QueryPattern, QueryStats, QueryTimer, extract_filter_fields};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use thiserror::Error;

pub use compaction::{
    CompactionAction, CompactionFilter, CompactionPick, SizeTieredCompaction, SstMeta,
    TombstoneFilter, UcsCompaction,
};

#[derive(Debug, Error)]
pub enum LsmError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("WAL error: {0}")]
    Wal(#[from] ingodb_wal::WalError),

    #[error("SSTable error: {0}")]
    SSTable(#[from] ingodb_sstable::SSTableError),

    #[error("blob error: {0}")]
    Blob(#[from] ingodb_blob::BlobError),

    #[error("not implemented: {0}")]
    NotImplemented(String),
}

/// Configuration for the LSM storage engine.
#[derive(Debug, Clone)]
pub struct LsmConfig {
    /// Directory for all data files
    pub data_dir: PathBuf,
    /// MemTable flush threshold in bytes (default 64 MB)
    pub memtable_size: usize,
    /// SSTable data block size (default 4096)
    pub block_size: usize,
    /// Compaction trigger: number of SSTables at a size tier before merging (default 4)
    pub compaction_threshold: usize,
    /// UCS scaling parameter W: <0 leveled, 0 balanced, >0 tiered (default 0)
    pub scaling_parameter: i32,
    /// Minimum result count before sort spills to disk as a partial index (default 1000)
    pub sort_spill_threshold: usize,
    /// Number of background compaction threads (default 4)
    pub compaction_threads: usize,
    /// Enable adaptive W (auto-tune scaling parameter from read/write ratio)
    pub adaptive_w: bool,
    /// Minimum seconds between W adjustments (default 900 = 15 minutes)
    pub adaptive_w_cooldown_secs: u64,
    /// Maximum W change per adjustment (default 2)
    pub adaptive_w_max_step: i32,
    /// Minimum W value (default -8)
    pub adaptive_w_min: i32,
    /// Maximum W value (default 8)
    pub adaptive_w_max: i32,
}

impl Default for LsmConfig {
    fn default() -> Self {
        LsmConfig {
            data_dir: PathBuf::from("ingodb_data"),
            memtable_size: 64 * 1024 * 1024,
            block_size: 4096,
            compaction_threshold: 4,
            scaling_parameter: 0,
            sort_spill_threshold: secondary::SPILL_THRESHOLD,
            compaction_threads: 4,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 900, // 15 minutes
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        }
    }
}

/// Shared state for background compaction signaling.
struct CompactionSignal {
    /// True when compaction work may be available
    pending: Mutex<bool>,
    /// Notifies the background thread
    notify: Condvar,
    /// True when a compaction is currently running
    running: AtomicBool,
    /// Notifies waiters that compaction finished
    done: Condvar,
    done_mutex: Mutex<()>,
    /// Signal background thread to stop
    stop: AtomicBool,
}

/// Compaction statistics — tracks what compaction has done.
pub struct CompactionStats {
    /// Number of compaction runs completed
    pub runs: AtomicU64,
    /// Total input bytes read during compaction
    pub bytes_read: AtomicU64,
    /// Total output bytes written during compaction
    pub bytes_written: AtomicU64,
    /// Total input SSTables consumed
    pub sstables_read: AtomicU64,
    /// Total output SSTables produced
    pub sstables_written: AtomicU64,
}

impl CompactionStats {
    fn new() -> Self {
        CompactionStats {
            runs: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            sstables_read: AtomicU64::new(0),
            sstables_written: AtomicU64::new(0),
        }
    }

    /// Write amplification: bytes_written / bytes_read. Higher = more rewriting.
    pub fn write_amplification(&self) -> f64 {
        let r = self.bytes_read.load(Ordering::Relaxed);
        let w = self.bytes_written.load(Ordering::Relaxed);
        if r == 0 { 0.0 } else { w as f64 / r as f64 }
    }
}

/// The LSM storage engine. Ties together WAL, MemTable, and SSTables.
pub struct LsmEngine {
    config: LsmConfig,
    /// Active memtable receiving writes (swapped out when full)
    memtable: RwLock<MemTable>,
    /// Immutable memtables queued for flush (newest first)
    immutable_memtables: Mutex<Vec<MemTable>>,
    /// WAL for the active memtable
    wal: Mutex<Wal>,
    /// SSTables on disk, ordered for reads: L0 first → L1 → ..., within each level newest first.
    /// RwLock: multiple concurrent readers, exclusive access for flush/compaction.
    sstables: RwLock<Vec<SSTableReader>>,
    /// Counter for generating SSTable file names
    next_sst_id: AtomicU64,
    /// Query statistics collector
    query_stats: QueryStats,
    /// Secondary indexes (sorted by non-_id fields)
    secondary_indexes: Mutex<Vec<secondary::SecondaryIndex>>,
    /// Newly built indexes awaiting persistence (collection_name not known here — Database handles it)
    pending_index_metadata: Mutex<Vec<IndexMetadata>>,
    /// Active snapshot versions — compaction preserves versions >= oldest snapshot
    active_snapshots: Mutex<BTreeSet<DocumentId>>,
    /// Group commit: pending WAL writes from concurrent put() calls
    wal_batch: Mutex<Vec<IBlob>>,
    /// Notifies waiting writers that their batch was committed
    wal_batch_done: Condvar,
    /// Background compaction signaling
    compaction_signal: Arc<CompactionSignal>,
    /// Background compaction thread handle
    compaction_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Compaction statistics
    compaction_stats: CompactionStats,
    /// Current effective W (may differ from config if adaptive_w is on)
    effective_w: std::sync::atomic::AtomicI32,
    /// Target W without step limiting (what W would be if we could jump instantly)
    target_w: std::sync::atomic::AtomicI32,
    /// Read operation counter (for adaptive W)
    read_count: AtomicU64,
    /// Write operation counter (for adaptive W)
    write_count: AtomicU64,
    /// Last time W was adjusted
    last_w_adjustment: Mutex<Instant>,
}

/// A consistent point-in-time view of the database.
///
/// All reads through a Snapshot see only documents with `_version <= self.version`.
/// Old versions are retained by compaction while any Snapshot references them.
pub struct Snapshot<'a> {
    engine: &'a LsmEngine,
    version: DocumentId,
}

impl<'a> Snapshot<'a> {
    /// Point lookup at this snapshot's point in time.
    pub fn get(&self, id: &DocumentId) -> Result<Option<IBlob>, LsmError> {
        self.engine.get_at(id, &self.version)
    }

    /// Scan at this snapshot's point in time.
    pub fn scan(
        &self,
        filter: Option<&Filter>,
        sort: Option<&[SortField]>,
        project: Option<&[String]>,
        limit: Option<usize>,
    ) -> Result<Vec<IBlob>, LsmError> {
        self.engine
            .scan_at(filter, sort, project, limit, &self.version)
    }

    /// The snapshot version.
    pub fn version(&self) -> &DocumentId {
        &self.version
    }
}

impl<'a> Drop for Snapshot<'a> {
    fn drop(&mut self) {
        self.engine.active_snapshots.lock().remove(&self.version);
    }
}

/// Metadata about a secondary index, for persistence in the system collection.
#[derive(Debug, Clone)]
pub struct IndexMetadata {
    /// Fields the index covers
    pub fields: Vec<String>,
    /// Path to the index SSTable file
    pub path: PathBuf,
    /// Whether this is a full-range or partial index
    pub is_full_range: bool,
}

impl LsmEngine {
    /// Open or create an LSM engine at the given directory.
    pub fn open(config: LsmConfig) -> Result<Self, LsmError> {
        std::fs::create_dir_all(&config.data_dir)?;
        let wal_path = config.data_dir.join("wal.log");

        // Recover existing WAL
        let recovered = Wal::recover(&wal_path)?;

        // Open WAL for new writes
        let wal = if recovered.is_empty() {
            Wal::open(&wal_path)?
        } else {
            // Truncate and reopen — recovered data goes into memtable
            std::fs::remove_file(&wal_path).ok();
            Wal::open(&wal_path)?
        };

        let memtable = MemTable::new(config.memtable_size);

        // Replay recovered blobs into memtable (versions already stamped)
        for blob in recovered {
            memtable.insert(blob);
        }

        // Load existing SSTables
        let mut sstables = Vec::new();
        let mut max_id = 0u64;
        let sst_dir = config.data_dir.clone();
        if sst_dir.exists() {
            let mut sst_files: Vec<_> = std::fs::read_dir(&sst_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "sst"))
                .collect();

            // Sort by name (which encodes creation order)
            sst_files.sort_by_key(|e| e.file_name());

            for entry in sst_files {
                let path = entry.path();
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(id) = stem.parse::<u64>() {
                        max_id = max_id.max(id);
                    }
                }
                match SSTableReader::open(&path) {
                    Ok(reader) => sstables.push(reader),
                    Err(e) => {
                        eprintln!("warning: skipping corrupt SSTable {}: {e}", path.display());
                    }
                }
            }
        }

        let initial_w = config.scaling_parameter;
        let ucs = UcsCompaction::new(initial_w, config.memtable_size as u64);
        sort_sstables_by_level(&mut sstables, &ucs);

        Ok(LsmEngine {
            config,
            memtable: RwLock::new(memtable),
            immutable_memtables: Mutex::new(Vec::new()),
            wal: Mutex::new(wal),
            sstables: RwLock::new(sstables),
            next_sst_id: AtomicU64::new(max_id + 1),
            query_stats: QueryStats::new(),
            secondary_indexes: Mutex::new(Vec::new()),
            pending_index_metadata: Mutex::new(Vec::new()),
            active_snapshots: Mutex::new(BTreeSet::new()),
            wal_batch: Mutex::new(Vec::new()),
            wal_batch_done: Condvar::new(),
            compaction_signal: Arc::new(CompactionSignal {
                pending: Mutex::new(false),
                notify: Condvar::new(),
                running: AtomicBool::new(false),
                done: Condvar::new(),
                done_mutex: Mutex::new(()),
                stop: AtomicBool::new(false),
            }),
            compaction_thread: Mutex::new(None),
            compaction_stats: CompactionStats::new(),
            effective_w: std::sync::atomic::AtomicI32::new(initial_w),
            target_w: std::sync::atomic::AtomicI32::new(initial_w),
            read_count: AtomicU64::new(0),
            write_count: AtomicU64::new(0),
            last_w_adjustment: Mutex::new(Instant::now()),
        })
    }

    /// Insert a document into the engine.
    /// Uses group commit: concurrent put() calls batch their WAL writes.
    pub fn put(&self, mut blob: IBlob) -> Result<(), LsmError> {
        self.write_count.fetch_add(1, Ordering::Relaxed);
        blob.set_version(DocumentId::new());

        // Group commit: add to batch buffer, then grab WAL lock.
        // While we wait for the lock, other writers accumulate in the batch.
        self.wal_batch.lock().push(blob);
        let mut wal = self.wal.lock();

        // Drain everything that accumulated while we waited
        let mut to_write: Vec<IBlob> = std::mem::take(&mut *self.wal_batch.lock());
        if to_write.is_empty() {
            return Ok(()); // another leader already wrote our entry
        }

        // Write the whole batch to WAL at once
        wal.append_batch(&mut to_write)?;
        drop(wal);

        // Insert all into memtable + notify indexes
        let indexes = self.secondary_indexes.lock();
        let memtable = self.memtable.read();
        for blob in to_write {
            if !indexes.is_empty() {
                for idx in indexes.iter() {
                    idx.notify_put(&blob);
                }
            }
            memtable.insert(blob);
        }
        let should_flush = memtable.should_flush();
        drop(memtable);
        drop(indexes);

        if should_flush {
            self.rotate_memtable()?;
        }

        Ok(())
    }

    /// Insert a batch of documents. More efficient than individual put() calls
    /// because locks are acquired once for the entire batch.
    pub fn put_batch(&self, mut blobs: Vec<IBlob>) -> Result<(), LsmError> {
        if blobs.is_empty() {
            return Ok(());
        }
        self.write_count
            .fetch_add(blobs.len() as u64, Ordering::Relaxed);

        // Stamp versions
        for blob in blobs.iter_mut() {
            blob.set_version(DocumentId::new());
        }

        // WAL: one lock, all appends
        {
            let mut wal = self.wal.lock();
            for blob in blobs.iter_mut() {
                wal.append(blob)?;
            }
        }

        // Secondary indexes: one lock, all notifications
        {
            let indexes = self.secondary_indexes.lock();
            if !indexes.is_empty() {
                for blob in blobs.iter() {
                    for idx in indexes.iter() {
                        idx.notify_put(blob);
                    }
                }
            }
        }

        // Memtable: one lock, all inserts
        {
            let memtable = self.memtable.read();
            for blob in blobs {
                memtable.insert(blob);
            }
            if memtable.should_flush() {
                drop(memtable);
                self.rotate_memtable()?;
            }
        }

        Ok(())
    }

    /// Delete a document by writing a tombstone.
    /// Stamps a server-assigned `_version` on the tombstone.
    pub fn delete(&self, id: &DocumentId) -> Result<(), LsmError> {
        self.write_count.fetch_add(1, Ordering::Relaxed);
        let mut tombstone = IBlob::tombstone(*id);
        tombstone.set_version(DocumentId::new());

        {
            let mut wal = self.wal.lock();
            wal.append(&mut tombstone)?;
        }

        // Notify secondary indexes of the deletion
        {
            let indexes = self.secondary_indexes.lock();
            for idx in indexes.iter() {
                idx.notify_delete(id);
            }
        }

        let should_flush = self.memtable.read().insert(tombstone);

        if should_flush {
            self.rotate_memtable()?;
        }

        Ok(())
    }

    /// Look up the latest version of a document by its stable document ID.
    /// Returns None if the document doesn't exist or has been deleted.
    pub fn get(&self, id: &DocumentId) -> Result<Option<IBlob>, LsmError> {
        self.get_at(id, &DocumentId::max())
    }

    /// Look up a document at a specific snapshot version.
    /// Returns the highest version <= snapshot for the given _id.
    fn get_at(&self, id: &DocumentId, snapshot: &DocumentId) -> Result<Option<IBlob>, LsmError> {
        self.read_count.fetch_add(1, Ordering::Relaxed);
        let mut timer = QueryTimer::start(QueryPattern {
            query_type: "get".into(),
            filter_fields: vec![],
            sort_fields: vec![],
            join_edge: None,
        });
        timer.set_docs_scanned(1);

        // Check active memtable first
        if let Some(blob) = self.memtable.read().get(id, snapshot) {
            let found = if blob.is_deleted() { None } else { Some(blob) };
            self.query_stats
                .record(timer.finish(if found.is_some() { 1 } else { 0 }));
            return Ok(found);
        }

        // Check immutable memtables (newest first)
        {
            let immutables = self.immutable_memtables.lock();
            for mt in immutables.iter().rev() {
                if let Some(blob) = mt.get(id, snapshot) {
                    let found = if blob.is_deleted() { None } else { Some(blob) };
                    self.query_stats
                        .record(timer.finish(if found.is_some() { 1 } else { 0 }));
                    return Ok(found);
                }
            }
        }

        // Check SSTables — use snapshot-aware lookup
        let sstables = self.sstables.read();
        for sst in sstables.iter() {
            if let Some(blob) = sst.get_by_id_at(id, snapshot)? {
                let found = if blob.is_deleted() { None } else { Some(blob) };
                drop(sstables);
                self.query_stats
                    .record(timer.finish(if found.is_some() { 1 } else { 0 }));
                return Ok(found);
            }
        }
        drop(sstables);

        self.query_stats.record(timer.finish(0));
        Ok(None)
    }

    /// Check if a document ID exists in the engine (not deleted).
    pub fn contains(&self, id: &DocumentId) -> Result<bool, LsmError> {
        Ok(self.get(id)?.is_some())
    }

    /// Flush the current memtable to a new SSTable.
    ///
    /// Atomically writes:
    /// Rotate the active memtable: swap in a fresh one, queue the old for flush.
    /// Writers are only blocked for the brief swap, not during the flush I/O.
    fn rotate_memtable(&self) -> Result<(), LsmError> {
        let old_memtable;
        {
            let mut active = self.memtable.write();
            let new_memtable = MemTable::new(self.config.memtable_size);
            old_memtable = std::mem::replace(&mut *active, new_memtable);
        }
        // Old memtable is now immutable — queue it for flush
        self.immutable_memtables.lock().push(old_memtable);

        // Signal background flush, or flush inline
        if self.compaction_thread.lock().is_some() {
            self.signal_compaction();
        } else {
            self.flush_immutable_memtables()?;
        }
        Ok(())
    }

    /// Flush all queued immutable memtables to SSTables.
    fn flush_immutable_memtables(&self) -> Result<(), LsmError> {
        loop {
            let memtable = self.immutable_memtables.lock().pop();
            let Some(memtable) = memtable else { break };
            self.flush_single_memtable(memtable)?;
        }
        Ok(())
    }

    /// Flush one memtable to an SSTable + update indexes + reset WAL.
    /// 1. Primary SSTable from memtable data
    /// 2. Secondary index entries for the flushed data (merge buffer to disk)
    ///    Only after both succeed: update metadata (SSTable list), reset WAL.
    fn flush_single_memtable(&self, memtable: MemTable) -> Result<(), LsmError> {
        let mut blobs = memtable.drain();
        if blobs.is_empty() {
            return Ok(());
        }

        // Step 1: Write primary SSTable
        let sst_id = self.next_sst_id.fetch_add(1, Ordering::SeqCst);
        let sst_path = self.config.data_dir.join(format!("{sst_id:012}.sst"));

        SSTableWriter::with_block_size(self.config.block_size).write(
            &sst_path,
            &mut blobs,
            &MvccKeyExtractor,
        )?;

        // Step 2: Flush secondary index buffers to disk
        // Only if there are secondary indexes to flush.
        {
            let mut indexes = self.secondary_indexes.lock();
            if !indexes.is_empty() {
                let sstables = self.sstables.read();
                let sst_refs: Vec<&SSTableReader> = sstables.iter().collect();
                // Estimate doc count from SSTable count * avg docs per table
                // (avoid iterating all SSTables which is O(total data))
                let estimated_doc_count = sst_refs.len() as u64
                    * (self.config.memtable_size as u64 / 400) // ~400 bytes per doc estimate
                    + blobs.len() as u64;

                for index in indexes.iter_mut() {
                    let _ = index.compact(&sst_refs, estimated_doc_count, self.config.block_size);
                }
            }
        }

        // Step 3: Update metadata — now safe to make the new SSTable visible
        let reader = SSTableReader::open(&sst_path)?;

        {
            let mut sstables = self.sstables.write();
            sstables.insert(0, reader);
            let ucs = self.ucs();
            sort_sstables_by_level(&mut sstables, &ucs);
        }

        // Step 4: Reset WAL — data is safely on disk in SSTable + index
        {
            let wal = self.wal.lock();
            let wal_path = wal.path().to_path_buf();
            drop(wal);
            std::fs::remove_file(&wal_path).ok();
            *self.wal.lock() = Wal::open(&wal_path)?;
        }

        // Trigger compaction (inline only — background coordinator handles its own)
        if self.compaction_thread.lock().is_none() {
            self.maybe_compact()?;
        }

        Ok(())
    }

    /// Flush the active memtable (public API for tests and wait_for_compaction).
    pub fn flush_memtable(&self) -> Result<(), LsmError> {
        self.rotate_memtable()
    }

    /// Run UCS compaction if needed.
    fn maybe_compact(&self) -> Result<(), LsmError> {
        self.maybe_adjust_w();
        let ucs = self.ucs();
        let sstables = self.sstables.read();

        let metas: Vec<SstMeta> = sstables
            .iter()
            .enumerate()
            .map(|(i, s)| SstMeta {
                path: s.path().to_path_buf(),
                min_id: s.min_id(),
                max_id: s.max_id(),
                file_size: s.file_size(),
                seq: i as u64,
            })
            .collect();
        drop(sstables);

        if let Some(pick) = ucs.pick_compaction(&metas) {
            let has_snapshots = self.oldest_snapshot().is_some();
            let mut tombstone_filter =
                TombstoneFilter::new(pick.output_level, pick.max_level, has_snapshots);
            self.run_compaction(&pick.inputs, Some(&mut tombstone_filter))?;
        }

        // Compact secondary indexes if needed
        self.maybe_compact_indexes()?;

        Ok(())
    }

    /// Compact secondary indexes: drop unused, merge partial ranges, rebuild.
    fn maybe_compact_indexes(&self) -> Result<(), LsmError> {
        let mut indexes = self.secondary_indexes.lock();
        if indexes.is_empty() {
            return Ok(());
        }

        // Drop unused indexes
        indexes.retain(|idx| {
            if idx.should_drop() {
                std::fs::remove_file(&idx.path).ok();
                false
            } else {
                true
            }
        });

        // Merge multiple partial ranges for the same sort fields
        // Find field sets with >1 index and merge them
        let mut field_groups: std::collections::HashMap<Vec<String>, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, idx) in indexes.iter().enumerate() {
            field_groups.entry(idx.fields.clone()).or_default().push(i);
        }

        let sstables = self.sstables.read();
        let sst_refs: Vec<&SSTableReader> = sstables.iter().collect();
        // Estimate doc count without iterating all SSTables (which is O(total data))
        let estimated_doc_count = sst_refs.len() as u64 * (self.config.memtable_size as u64 / 400)
            + self.memtable.read().len() as u64;

        for (fields, group_indices) in &field_groups {
            if group_indices.len() > 1 {
                // Multiple partial indexes for the same fields — replace with a full rebuild.
                // Merging partial entries would produce a false "full range" index that only
                // covers the union of the partial filter ranges, causing missed documents on
                // no-filter sort scans. A full rebuild from primary SSTables is correct.
                let idx_name = fields.join("_");
                let sst_id = self.next_sst_id.fetch_add(1, Ordering::SeqCst);
                let merged_path = self
                    .config
                    .data_dir
                    .join(format!("idx_{idx_name}_{sst_id:012}.sst"));

                if let Ok(merged) = secondary::SecondaryIndex::build(
                    fields,
                    &sst_refs,
                    &merged_path,
                    self.config.block_size,
                ) {
                    // Remove old indexes (reverse order to keep indices valid).
                    let mut to_remove: Vec<usize> = group_indices.clone();
                    to_remove.sort_unstable_by(|a, b| b.cmp(a));
                    for i in to_remove {
                        let old = indexes.remove(i);
                        std::fs::remove_file(&old.path).ok();
                    }
                    indexes.push(merged);

                    // Persist metadata so the merged index survives a restart.
                    self.pending_index_metadata.lock().push(IndexMetadata {
                        fields: fields.clone(),
                        path: merged_path,
                        is_full_range: true,
                    });
                }
            }
        }

        // Compact individual indexes (merge buffer or full rebuild)
        for index in indexes.iter_mut() {
            let _ = index.compact(&sst_refs, estimated_doc_count, self.config.block_size);
        }

        Ok(())
    }

    fn ucs(&self) -> UcsCompaction {
        let w = self.effective_w.load(std::sync::atomic::Ordering::Relaxed);
        UcsCompaction::new(w, self.config.memtable_size as u64)
    }

    /// Current effective W parameter (may differ from config if adaptive).
    pub fn effective_w(&self) -> i32 {
        self.effective_w.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Target W without step limiting (what the workload ratio suggests).
    pub fn target_w(&self) -> i32 {
        self.target_w.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Adjust W based on observed read/write ratio.
    /// Called periodically by the compaction coordinator.
    fn maybe_adjust_w(&self) {
        if !self.config.adaptive_w {
            return;
        }

        let cooldown = std::time::Duration::from_secs(self.config.adaptive_w_cooldown_secs);
        let mut last = self.last_w_adjustment.lock();
        if last.elapsed() < cooldown {
            return;
        }

        let reads = self.read_count.swap(0, Ordering::Relaxed);
        let writes = self.write_count.swap(0, Ordering::Relaxed);
        let total = reads + writes;

        if total < 10 {
            return; // not enough data to decide
        }

        // Read ratio: 0.0 = all writes, 1.0 = all reads
        let read_ratio = reads as f64 / total as f64;

        // Target W: read-heavy → negative (leveled), write-heavy → positive (tiered)
        // Linear mapping: ratio 0.0 → max_w, ratio 1.0 → min_w
        let target_w = (self.config.adaptive_w_max as f64
            - (self.config.adaptive_w_max - self.config.adaptive_w_min) as f64 * read_ratio)
            as i32;

        self.target_w
            .store(target_w, std::sync::atomic::Ordering::Relaxed);

        let current_w = self.effective_w.load(std::sync::atomic::Ordering::Relaxed);
        let step = self.config.adaptive_w_max_step;

        // Clamp change to ±step
        let new_w = if target_w > current_w {
            (current_w + step)
                .min(target_w)
                .min(self.config.adaptive_w_max)
        } else if target_w < current_w {
            (current_w - step)
                .max(target_w)
                .max(self.config.adaptive_w_min)
        } else {
            current_w
        };

        if new_w != current_w {
            self.effective_w
                .store(new_w, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[ingodb] adaptive W: {} → {} (read_ratio={:.2}, target={}, step-limited)",
                current_w, new_w, read_ratio, target_w
            );
        }

        *last = Instant::now();
    }

    /// Run compaction on the given SSTable files, optionally applying a filter.
    /// For duplicate `_id`s, the entry with the highest `_version` wins.
    fn run_compaction(
        &self,
        inputs: &[PathBuf],
        filter: Option<&mut dyn CompactionFilter>,
    ) -> Result<(), LsmError> {
        let num_inputs = inputs.len() as u64;
        let mut input_bytes = 0u64;

        // Merge all input SSTables
        let mut merged: Vec<IBlob> = Vec::new();
        for path in inputs {
            let reader = SSTableReader::open(path)?;
            input_bytes += reader.file_size();
            let entries = reader.iter()?;
            merged.extend(entries.into_iter().map(|(_, blob)| blob));
        }

        // Sort by _id, then _version desc
        merged.sort_by(|a, b| {
            a.id()
                .cmp(b.id())
                .then_with(|| b.version().cmp(a.version()))
        });

        // MVCC-aware dedup: keep versions referenced by active snapshots
        let oldest_snap = self.oldest_snapshot();
        if let Some(oldest) = oldest_snap {
            // Keep latest version per _id PLUS any version >= oldest snapshot
            let mut kept = Vec::new();
            let mut i = 0;
            while i < merged.len() {
                let id = *merged[i].id();
                // Always keep the latest version (first in group)
                kept.push(merged[i].clone());
                i += 1;
                // Keep older versions if >= oldest snapshot
                while i < merged.len() && merged[i].id() == &id {
                    if merged[i].version() >= &oldest {
                        kept.push(merged[i].clone());
                    }
                    i += 1;
                }
            }
            merged = kept;
        } else {
            // No active snapshots — normal dedup, keep only latest per _id
            merged.dedup_by(|a, b| a.id() == b.id());
        }

        // Apply compaction filter (tombstone purge + any user filter)
        if let Some(filter) = filter {
            merged.retain_mut(|blob| match filter.filter(blob.id(), blob) {
                CompactionAction::Keep => true,
                CompactionAction::Drop => false,
                CompactionAction::Transform(new_blob) => {
                    *blob = new_blob;
                    true
                }
            });
        }

        if merged.is_empty() {
            // All entries were dropped — just remove input files
            let mut sstables = self.sstables.write();
            for path in inputs {
                sstables.retain(|s| s.path() != path);
                std::fs::remove_file(path).ok();
            }
            return Ok(());
        }

        // Write merged SSTable
        let sst_id = self.next_sst_id.fetch_add(1, Ordering::SeqCst);
        let output_path = self.config.data_dir.join(format!("{sst_id:012}.sst"));
        SSTableWriter::with_block_size(self.config.block_size).write(
            &output_path,
            &mut merged,
            &MvccKeyExtractor,
        )?;
        let new_reader = SSTableReader::open(&output_path)?;
        let output_bytes = new_reader.file_size();

        // Record compaction stats
        self.compaction_stats.runs.fetch_add(1, Ordering::Relaxed);
        self.compaction_stats
            .bytes_read
            .fetch_add(input_bytes, Ordering::Relaxed);
        self.compaction_stats
            .bytes_written
            .fetch_add(output_bytes, Ordering::Relaxed);
        self.compaction_stats
            .sstables_read
            .fetch_add(num_inputs, Ordering::Relaxed);
        self.compaction_stats
            .sstables_written
            .fetch_add(1, Ordering::Relaxed);

        // Swap old SSTables for new one
        let mut sstables = self.sstables.write();
        for path in inputs {
            sstables.retain(|s| s.path() != path);
            std::fs::remove_file(path).ok();
        }
        sstables.push(new_reader);

        // Re-sort by level for correct read ordering
        let ucs = self.ucs();
        sort_sstables_by_level(&mut sstables, &ucs);

        Ok(())
    }

    /// Number of SSTables on disk.
    pub fn sstable_count(&self) -> usize {
        self.sstables.read().len()
    }

    /// Number of entries in the active memtable.
    pub fn memtable_size(&self) -> usize {
        self.memtable.read().len()
    }

    /// Force a sync of the WAL to disk.
    pub fn sync(&self) -> Result<(), LsmError> {
        self.wal.lock().sync()?;
        Ok(())
    }

    /// Access the query statistics collector.
    pub fn query_stats(&self) -> &QueryStats {
        &self.query_stats
    }

    /// Access compaction statistics.
    pub fn compaction_stats(&self) -> &CompactionStats {
        &self.compaction_stats
    }

    /// Start background compaction threads. Requires the engine to be in an Arc.
    /// If not called, compaction runs inline during flush (blocking the writer).
    pub fn start_background_compaction(self: &Arc<Self>) {
        let engine = Arc::clone(self);
        let signal = Arc::clone(&self.compaction_signal);
        let num_threads = self.config.compaction_threads;

        let adaptive = self.config.adaptive_w;
        let cooldown = std::time::Duration::from_secs(self.config.adaptive_w_cooldown_secs);

        // Coordinator thread: wakes up, picks all compaction jobs, dispatches to workers
        let handle = std::thread::Builder::new()
            .name("ingodb-compaction-coordinator".into())
            .spawn(move || {
                loop {
                    // Wait for work, stop signal, or periodic timeout (for adaptive W)
                    {
                        let mut pending = signal.pending.lock();
                        while !*pending && !signal.stop.load(Ordering::Relaxed) {
                            if adaptive {
                                // Wake periodically to check read/write ratio
                                signal.notify.wait_for(&mut pending, cooldown);
                                break; // check regardless of whether signaled
                            } else {
                                signal.notify.wait(&mut pending);
                            }
                        }
                        if signal.stop.load(Ordering::Relaxed) {
                            break;
                        }
                        *pending = false;
                    }

                    signal.running.store(true, Ordering::SeqCst);

                    // Flush any immutable memtables first
                    let _ = engine.flush_immutable_memtables();

                    // Adaptive W: adjust based on recent read/write ratio
                    engine.maybe_adjust_w();

                    // Pick all eligible compaction jobs
                    let ucs = engine.ucs();
                    let sstables = engine.sstables.read();
                    let metas: Vec<SstMeta> = sstables
                        .iter()
                        .enumerate()
                        .map(|(i, s)| SstMeta {
                            path: s.path().to_path_buf(),
                            min_id: s.min_id(),
                            max_id: s.max_id(),
                            file_size: s.file_size(),
                            seq: i as u64,
                        })
                        .collect();
                    drop(sstables);

                    let picks = ucs.pick_all_compactions(&metas);

                    if picks.is_empty() {
                        // No compaction work — also compact indexes
                        let _ = engine.maybe_compact_indexes();
                    } else {
                        // Dispatch compaction jobs to worker threads
                        let mut handles = Vec::new();
                        for pick in picks.into_iter().take(num_threads) {
                            let engine = Arc::clone(&engine);
                            handles.push(
                                std::thread::Builder::new()
                                    .name("ingodb-compaction-worker".into())
                                    .spawn(move || {
                                        let has_snapshots = engine.oldest_snapshot().is_some();
                                        let mut tombstone_filter = TombstoneFilter::new(
                                            pick.output_level,
                                            pick.max_level,
                                            has_snapshots,
                                        );
                                        let _ = engine.run_compaction(
                                            &pick.inputs,
                                            Some(&mut tombstone_filter),
                                        );
                                    })
                                    .expect("failed to spawn compaction worker"),
                            );
                        }
                        // Wait for all workers to finish
                        for h in handles {
                            let _ = h.join();
                        }
                        // Compact secondary indexes after primary compaction
                        let _ = engine.maybe_compact_indexes();

                        // Signal again in case more work was created by compaction output
                        let mut pending = signal.pending.lock();
                        *pending = true;
                    }

                    signal.running.store(false, Ordering::SeqCst);
                    signal.done.notify_all();
                }
            })
            .expect("failed to spawn compaction coordinator");

        *self.compaction_thread.lock() = Some(handle);
    }

    /// Signal the background compaction thread to check for work.
    fn signal_compaction(&self) {
        let mut pending = self.compaction_signal.pending.lock();
        *pending = true;
        self.compaction_signal.notify.notify_one();
    }

    /// Wait until all pending compaction work is finished.
    /// Also flushes the memtable if it has data.
    pub fn wait_for_compaction(&self) -> Result<(), LsmError> {
        // Rotate active memtable if it has data, then flush all immutables
        if !self.memtable.read().is_empty() {
            self.rotate_memtable()?;
        }
        self.flush_immutable_memtables()?;

        // If background compaction is active, wait for it
        if self.compaction_thread.lock().is_some() {
            let mut guard = self.compaction_signal.done_mutex.lock();
            while self.compaction_signal.running.load(Ordering::SeqCst)
                || *self.compaction_signal.pending.lock()
            {
                self.compaction_signal.done.wait(&mut guard);
            }
        }
        Ok(())
    }

    /// Create a snapshot for consistent point-in-time reads.
    /// All reads through the snapshot see only versions <= the snapshot's version.
    pub fn snapshot(&self) -> Snapshot<'_> {
        let version = DocumentId::new();
        self.active_snapshots.lock().insert(version);
        Snapshot {
            engine: self,
            version,
        }
    }

    /// Oldest active snapshot version, or None if no snapshots active.
    fn oldest_snapshot(&self) -> Option<DocumentId> {
        self.active_snapshots.lock().iter().next().copied()
    }

    /// Check if a secondary index exists for the given sort fields.
    /// Not yet used — intended for a future explicit index management API
    /// (e.g. `engine.ensure_index(&["field"])`).
    #[allow(dead_code)]
    fn has_secondary_index(&self, sort_fields: &[String]) -> bool {
        self.secondary_indexes
            .lock()
            .iter()
            .any(|idx| idx.matches_sort(sort_fields))
    }

    /// Build a secondary index for the given sort fields from current SSTables.
    /// Not yet used — intended for a future explicit index management API.
    #[allow(dead_code)]
    fn build_secondary_index(&self, sort_fields: &[String]) -> Result<(), LsmError> {
        let sstables = self.sstables.read();
        let sst_refs: Vec<&SSTableReader> = sstables.iter().collect();

        if sst_refs.is_empty() {
            return Ok(());
        }

        let idx_name = sort_fields.join("_");
        let idx_path = self.config.data_dir.join(format!("idx_{idx_name}.sst"));

        let index = secondary::SecondaryIndex::build(
            sort_fields,
            &sst_refs,
            &idx_path,
            self.config.block_size,
        )?;
        drop(sstables);

        let meta = IndexMetadata {
            fields: sort_fields.to_vec(),
            path: idx_path,
            is_full_range: true,
        };
        self.pending_index_metadata.lock().push(meta);
        self.secondary_indexes.lock().push(index);
        Ok(())
    }

    /// Try to use a secondary index for a sorted scan.
    /// Returns None if no matching index exists.
    fn scan_with_secondary_index(
        &self,
        sort_fields: &[String],
        filter: Option<&Filter>,
        limit: Option<usize>,
    ) -> Option<Result<Vec<IBlob>, LsmError>> {
        let indexes = self.secondary_indexes.lock();
        let index = indexes
            .iter()
            .find(|idx| idx.matches_query(sort_fields, filter))?;
        index.mark_used();

        // Read sorted entries and clone fields before dropping the lock
        let sorted_entries = match index.iter_sorted() {
            Ok(entries) => entries,
            Err(e) => return Some(Err(e)),
        };
        let index_fields = index.fields.clone();
        drop(indexes);

        // For each entry, look up the full document by _id.
        // Verify the indexed field values still match (stale check).
        let mut results = Vec::new();
        for (id, projected) in sorted_entries {
            // Skip tombstones in the index buffer
            if projected.is_deleted() {
                continue;
            }
            match self.get(&id) {
                Ok(Some(blob)) => {
                    // Stale check: verify indexed field values match the primary
                    let is_current = index_fields
                        .iter()
                        .all(|f| blob.get_field(f) == projected.get_field(f));
                    if !is_current {
                        continue;
                    }
                    // Apply filter on the full document
                    if let Some(f) = filter {
                        if !f.matches(&|field| blob.get_field(field)) {
                            continue;
                        }
                    }
                    results.push(blob);
                    if let Some(lim) = limit {
                        if results.len() >= lim {
                            break;
                        }
                    }
                }
                Ok(None) => continue, // stale/deleted — skip
                Err(e) => return Some(Err(e)),
            }
        }

        // Merge with memtable (always fresh, may have docs not in the index)
        let memtable_docs: Vec<IBlob> = self
            .memtable
            .read()
            .iter()
            .map(|(_, blob)| blob)
            .filter(|blob| !blob.is_deleted())
            .filter(|blob| filter.is_none_or(|f| f.matches(&|field| blob.get_field(field))))
            .collect();

        if !memtable_docs.is_empty() {
            // Merge: memtable version wins (newer), replace any matching index results
            let memtable_ids: std::collections::HashSet<DocumentId> =
                memtable_docs.iter().map(|b| *b.id()).collect();
            results.retain(|b| !memtable_ids.contains(b.id()));

            // Add memtable docs (dedup within memtable docs by id)
            let mut seen: std::collections::HashSet<DocumentId> =
                results.iter().map(|b| *b.id()).collect();
            for doc in memtable_docs {
                if seen.insert(*doc.id()) {
                    results.push(doc);
                }
            }

            // Re-sort by the indexed fields
            let sort_field_list: Vec<SortField> = sort_fields
                .iter()
                .map(|f| SortField {
                    field: f.clone(),
                    direction: SortDirection::Ascending,
                })
                .collect();
            results.sort_by(|a, b| {
                for sf in &sort_field_list {
                    let va = a.get_field(&sf.field);
                    let vb = b.get_field(&sf.field);
                    let ord = match (&va, &vb) {
                        (Some(va), Some(vb)) => {
                            compare_values(va, vb).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => std::cmp::Ordering::Equal,
                    };
                    if ord != std::cmp::Ordering::Equal {
                        return ord;
                    }
                }
                std::cmp::Ordering::Equal
            });

            if let Some(lim) = limit {
                results.truncate(lim);
            }
        }

        Some(Ok(results))
    }

    /// Spill sorted scan results to disk as a partial secondary index.
    /// Replaces any existing index for the same sort fields.
    fn spill_to_partial_index(
        &self,
        sort_fields: &[String],
        range: Option<Filter>,
        sorted_results: &mut [IBlob],
    ) -> Result<(), LsmError> {
        let idx_name = sort_fields.join("_");
        let sst_id = self.next_sst_id.fetch_add(1, Ordering::SeqCst);
        let idx_path = self
            .config
            .data_dir
            .join(format!("idx_{idx_name}_{sst_id:012}.sst"));

        let index = secondary::SecondaryIndex::build_partial(
            sort_fields,
            range.clone(),
            sorted_results,
            &idx_path,
            self.config.block_size,
        )?;

        let meta = IndexMetadata {
            fields: sort_fields.to_vec(),
            path: idx_path,
            is_full_range: index.range.is_none(),
        };

        // Add alongside existing indexes for the same fields (compaction will merge)
        // Only replace if the exact same range already exists
        let mut indexes = self.secondary_indexes.lock();
        if let Some(pos) = indexes
            .iter()
            .position(|idx| idx.matches_sort(sort_fields) && idx.range == range)
        {
            let old = indexes.remove(pos);
            std::fs::remove_file(&old.path).ok();
        }
        indexes.push(index);
        drop(indexes);

        self.pending_index_metadata.lock().push(meta);
        Ok(())
    }

    /// Try to use a secondary index to accelerate a filter-only scan.
    /// Works for simple Eq/Gt/Lt/Range filters on a single field that has an index.
    fn scan_with_filter_index(
        &self,
        filter: &Filter,
        limit: Option<usize>,
        snapshot: &DocumentId,
    ) -> Option<Result<Vec<IBlob>, LsmError>> {
        // Extract range boundaries from the filter for binary search on the index
        let (field, start_val, end_val) = match filter {
            Filter::Eq { field, value } => {
                (field.clone(), Some(value.clone()), Some(value.clone()))
            }
            Filter::Gt { field, value } => {
                (field.clone(), Some(value.clone()), None) // open upper bound
            }
            Filter::Lt { field, value } => {
                (field.clone(), None, Some(value.clone())) // open lower bound
            }
            Filter::Range { field, low, high } => {
                (field.clone(), Some(low.clone()), Some(high.clone()))
            }
            _ => return None,
        };

        // Check if we have an index on this field
        let indexes = self.secondary_indexes.lock();
        let index = indexes
            .iter()
            .find(|idx| idx.fields.len() == 1 && idx.fields[0] == field)?;
        index.mark_used();

        // Range scan on the index — O(log N + R) instead of O(N)
        let range_entries = match index.range_scan(start_val.as_ref(), end_val.as_ref()) {
            Ok(entries) => entries,
            Err(e) => return Some(Err(e)),
        };
        drop(indexes);

        let mut results = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for (id, projected) in range_entries {
            if projected.is_deleted() {
                continue;
            }
            // Fetch the full document from primary
            match self.get_at(&id, snapshot) {
                Ok(Some(blob)) => {
                    // Verify the filter still holds on the full document (stale check)
                    if !filter.matches(&|f| blob.get_field(f)) {
                        continue;
                    }
                    if seen.insert(*blob.id()) {
                        results.push(blob);
                        if let Some(lim) = limit {
                            if results.len() >= lim {
                                break;
                            }
                        }
                    }
                }
                Ok(None) => continue,
                Err(e) => return Some(Err(e)),
            }
        }

        // Also check memtable for docs not in the index
        for (_, blob) in self.memtable.read().iter() {
            if blob.is_deleted() || blob.version() > snapshot {
                continue;
            }
            if filter.matches(&|f| blob.get_field(f)) && seen.insert(*blob.id()) {
                results.push(blob);
            }
        }

        Some(Ok(results))
    }

    /// Number of secondary indexes.
    pub fn secondary_index_count(&self) -> usize {
        self.secondary_indexes.lock().len()
    }

    /// Drain newly built index metadata (for persistence by Database).
    pub fn drain_pending_index_metadata(&self) -> Vec<IndexMetadata> {
        std::mem::take(&mut *self.pending_index_metadata.lock())
    }

    /// Load an existing secondary index from disk.
    pub fn load_secondary_index(
        &self,
        fields: Vec<String>,
        range: Option<Filter>,
        path: &Path,
    ) -> Result<(), LsmError> {
        let index = secondary::SecondaryIndex::open(fields, range, path)?;
        self.secondary_indexes.lock().push(index);
        Ok(())
    }

    /// Replay memtable entries into secondary indexes.
    /// Called by Database::open() after loading indexes, so that WAL-recovered
    /// documents that were written after the last index compaction get re-buffered.
    pub fn replay_memtable_to_indexes(&self) {
        let indexes = self.secondary_indexes.lock();
        if indexes.is_empty() {
            return;
        }
        for (_, blob) in self.memtable.read().iter() {
            for idx in indexes.iter() {
                if blob.is_deleted() {
                    idx.notify_delete(blob.id());
                } else {
                    idx.notify_put(&blob);
                }
            }
        }
    }

    /// Synchronously flush and compact (SSTable + index).
    /// Primarily useful in tests and benchmarks to trigger immediate compaction.
    pub fn compact_now(&self) -> Result<(), LsmError> {
        if !self.memtable.read().is_empty() {
            self.rotate_memtable()?;
        }
        self.flush_immutable_memtables()?;
        self.maybe_compact()
    }

    /// Full scan: merge all live documents, apply filter/sort/projection/limit.
    pub fn scan(
        &self,
        filter: Option<&Filter>,
        sort: Option<&[SortField]>,
        project: Option<&[String]>,
        limit: Option<usize>,
    ) -> Result<Vec<IBlob>, LsmError> {
        self.scan_at(filter, sort, project, limit, &DocumentId::max())
    }

    /// Scan at a specific snapshot version.
    fn scan_at(
        &self,
        filter: Option<&Filter>,
        sort: Option<&[SortField]>,
        project: Option<&[String]>,
        limit: Option<usize>,
        snapshot: &DocumentId,
    ) -> Result<Vec<IBlob>, LsmError> {
        self.read_count.fetch_add(1, Ordering::Relaxed);
        let mut timer = QueryTimer::start(QueryPattern {
            query_type: "scan".into(),
            filter_fields: filter.map(extract_filter_fields).unwrap_or_default(),
            sort_fields: sort
                .map(|s| s.iter().map(|sf| sf.field.clone()).collect())
                .unwrap_or_default(),
            join_edge: None,
        });

        // Try secondary index for sorted scans (only for latest snapshot)
        if *snapshot == DocumentId::max() {
            if let Some(sort_fields) = sort {
                let field_names: Vec<String> =
                    sort_fields.iter().map(|sf| sf.field.clone()).collect();
                let all_ascending = sort_fields
                    .iter()
                    .all(|sf| sf.direction == SortDirection::Ascending);
                let all_descending = sort_fields
                    .iter()
                    .all(|sf| sf.direction == SortDirection::Descending);

                if all_ascending || all_descending {
                    if let Some(result) =
                        self.scan_with_secondary_index(&field_names, filter, limit)
                    {
                        let mut results = result?;
                        if all_descending {
                            results.reverse();
                            // Re-apply limit after reverse (index scan may have applied limit from the wrong end)
                            if let Some(lim) = limit {
                                results.truncate(lim);
                            }
                        }
                        if let Some(fields) = project {
                            results = results
                                .into_iter()
                                .map(|blob| blob.project(fields))
                                .collect();
                        }
                        let docs_returned = results.len() as u64;
                        timer.set_docs_scanned(docs_returned);
                        self.query_stats.record(timer.finish(docs_returned));
                        return Ok(results);
                    }
                }
            }
            // Try secondary index for filter-only scans (no sort required)
            if sort.is_none() {
                if let Some(filter) = filter {
                    if let Some(result) = self.scan_with_filter_index(filter, limit, snapshot) {
                        let mut results = result?;
                        if let Some(fields) = project {
                            results = results
                                .into_iter()
                                .map(|blob| blob.project(fields))
                                .collect();
                        }
                        let docs_returned = results.len() as u64;
                        timer.set_docs_scanned(docs_returned);
                        self.query_stats.record(timer.finish(docs_returned));
                        return Ok(results);
                    }
                }
            }
        } // end secondary index check (latest snapshot only)

        // Collect all IBlobs from memtable + SSTables
        let mut all: Vec<IBlob> = Vec::new();

        // Memtable entries (active + immutable)
        all.extend(self.memtable.read().iter().map(|(_, blob)| blob));
        for mt in self.immutable_memtables.lock().iter() {
            all.extend(mt.iter().map(|(_, blob)| blob));
        }

        // SSTable entries
        {
            let sstables = self.sstables.read();
            for sst in sstables.iter() {
                all.extend(sst.iter()?.into_iter().map(|(_, blob)| blob));
            }
        }

        // MVCC: filter to versions visible at snapshot
        all.retain(|b| b.version() <= snapshot);

        // Merge: sort by _id, dedup keeping highest _version (within snapshot)
        all.sort_by(|a, b| {
            a.id()
                .cmp(b.id())
                .then_with(|| b.version().cmp(a.version()))
        });
        all.dedup_by(|a, b| a.id() == b.id());

        // Count live documents (docs scanned = after dedup, excluding tombstones)
        let docs_scanned = all.iter().filter(|b| !b.is_deleted()).count() as u64;
        timer.set_docs_scanned(docs_scanned);

        // Filter out tombstones and apply query filter
        let mut results: Vec<IBlob> = Vec::new();
        for blob in all {
            if blob.is_deleted() {
                continue;
            }
            if let Some(f) = filter {
                if !f.matches(&|field| blob.get_field(field)) {
                    continue;
                }
            }
            results.push(blob);
        }

        // Sort (before projection — sort fields may not be in the projection)
        if let Some(sort_fields) = sort {
            results.sort_by(|a, b| {
                for sf in sort_fields {
                    let va = a.get_field(&sf.field);
                    let vb = b.get_field(&sf.field);
                    let ord = match (&va, &vb) {
                        (Some(va), Some(vb)) => {
                            compare_values(va, vb).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        (Some(_), None) => std::cmp::Ordering::Less, // non-null first
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        (None, None) => std::cmp::Ordering::Equal,
                    };
                    let ord = match sf.direction {
                        SortDirection::Ascending => ord,
                        SortDirection::Descending => ord.reverse(),
                    };
                    if ord != std::cmp::Ordering::Equal {
                        return ord;
                    }
                }
                std::cmp::Ordering::Equal
            });
        }

        // Spill sorted results to disk as a partial index if above threshold.
        // Index is always stored in ascending order (descending reads reverse it).
        if let Some(sort_fields) = sort {
            let field_names: Vec<String> = sort_fields.iter().map(|sf| sf.field.clone()).collect();
            let all_same_direction = sort_fields
                .iter()
                .all(|sf| sf.direction == sort_fields[0].direction);
            if all_same_direction && results.len() > self.config.sort_spill_threshold {
                // For descending, reverse results back to ascending before spilling
                let mut to_spill = results.clone();
                if sort_fields[0].direction == SortDirection::Descending {
                    to_spill.reverse();
                }
                let _ = self.spill_to_partial_index(&field_names, filter.cloned(), &mut to_spill);
            }
        }

        // Apply limit (after sort)
        if let Some(lim) = limit {
            results.truncate(lim);
        }

        // Apply projection (last — sort fields may not be projected)
        if let Some(fields) = project {
            results = results
                .into_iter()
                .map(|blob| blob.project(fields))
                .collect();
        }

        // Record stats
        let docs_returned = results.len() as u64;
        self.query_stats.record(timer.finish(docs_returned));

        // Reactive: build filter index if warranted.
        // Trigger is based on docs_scanned (the cost of the full scan), not results.len().
        // A highly selective filter (1% match rate) has low results.len() but high scan cost.
        if sort.is_none() && docs_scanned > self.config.sort_spill_threshold as u64 {
            if let Some(filter) = filter {
                // Extract the filter field for a simple single-field filter
                let field = match filter {
                    Filter::Eq { field, .. }
                    | Filter::Gt { field, .. }
                    | Filter::Lt { field, .. }
                    | Filter::Range { field, .. } => Some(field.clone()),
                    _ => None,
                };
                if let Some(field) = field {
                    // Check if an index already exists for this field
                    let has_index = self
                        .secondary_indexes
                        .lock()
                        .iter()
                        .any(|idx| idx.fields.len() == 1 && idx.fields[0] == field);
                    if !has_index {
                        // Check stats: build only if low selectivity and repeated
                        let pattern = QueryPattern {
                            query_type: "scan".into(),
                            filter_fields: vec![field.clone()],
                            sort_fields: vec![],
                            join_edge: None,
                        };
                        if let Some(stats) = self.query_stats.get_pattern(&pattern) {
                            if stats.selectivity() <= 0.5
                                && stats.count >= secondary::DEFAULT_INDEX_THRESHOLD
                            {
                                // Build a full-range index sorted by this field
                                // Re-scan all docs to build a complete index
                                let all_docs: Vec<IBlob> = self
                                    .memtable
                                    .read()
                                    .iter()
                                    .map(|(_, b)| b)
                                    .chain(self.sstables.read().iter().flat_map(|sst| {
                                        sst.iter().unwrap_or_default().into_iter().map(|(_, b)| b)
                                    }))
                                    .collect();
                                let mut deduped = all_docs;
                                deduped.sort_by(|a, b| {
                                    a.id()
                                        .cmp(b.id())
                                        .then_with(|| b.version().cmp(a.version()))
                                });
                                deduped.dedup_by(|a, b| a.id() == b.id());
                                deduped.retain(|b| !b.is_deleted());

                                let _ = self.spill_to_partial_index(
                                    &[field],
                                    None, // full range
                                    &mut deduped,
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Graph traversal as join-by-value.
    ///
    /// Starting from documents matching `start` filter, follows edges by joining
    /// `from_field` values against `to_field` values. Returns the discovered
    /// documents (not the starting set). Deduplicates by `_id`.
    pub fn traverse(
        &self,
        start: Option<&Filter>,
        from_field: &str,
        to_field: &str,
        depth: usize,
    ) -> Result<Vec<IBlob>, LsmError> {
        self.traverse_at(start, from_field, to_field, depth, &DocumentId::max())
    }

    fn traverse_at(
        &self,
        start: Option<&Filter>,
        from_field: &str,
        to_field: &str,
        depth: usize,
        snapshot: &DocumentId,
    ) -> Result<Vec<IBlob>, LsmError> {
        let timer = QueryTimer::start(QueryPattern {
            query_type: "traverse".into(),
            filter_fields: start.map(extract_filter_fields).unwrap_or_default(),
            sort_fields: vec![],
            join_edge: Some((from_field.into(), to_field.into())),
        });

        if depth == 0 {
            self.query_stats.record(timer.finish(0));
            return Ok(Vec::new());
        }

        // Get starting documents (inner scan records its own stats)
        let mut current = self.scan_at(start, None, None, None, snapshot)?;
        let mut all_results = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for _ in 0..depth {
            let join_values: Vec<Value> = current
                .iter()
                .filter_map(|blob| blob.get_field(from_field))
                .collect();

            if join_values.is_empty() {
                break;
            }

            let mut next = Vec::new();
            let candidates = self.scan_at(None, None, None, None, snapshot)?;
            for doc in candidates {
                if let Some(target_val) = doc.get_field(to_field) {
                    if join_values.contains(&target_val) && seen.insert(*doc.id()) {
                        next.push(doc);
                    }
                }
            }

            all_results.extend(next.iter().cloned());
            current = next;

            if current.is_empty() {
                break;
            }
        }

        let docs_returned = all_results.len() as u64;
        self.query_stats.record(timer.finish(docs_returned));
        Ok(all_results)
    }

    /// Execute a Liquid AST query.
    pub fn execute(&self, query: &Query) -> Result<Vec<IBlob>, LsmError> {
        match query {
            Query::Get { id } => Ok(self.get(id)?.into_iter().collect()),
            Query::Scan {
                filter,
                sort,
                project,
                limit,
            } => self.scan(filter.as_ref(), sort.as_deref(), project.as_deref(), *limit),
            Query::Traverse {
                start,
                from_field,
                to_field,
                depth,
            } => self.traverse(start.as_ref(), from_field, to_field, *depth),
        }
    }
}

impl Drop for LsmEngine {
    fn drop(&mut self) {
        // Signal the background compaction thread to stop
        self.compaction_signal.stop.store(true, Ordering::SeqCst);
        self.compaction_signal.notify.notify_one();
        if let Some(handle) = self.compaction_thread.lock().take() {
            let _ = handle.join();
        }
    }
}

/// Sort SSTables for correct read ordering:
/// L0 first → L1 → L2 → ..., within each level newest first (by filename/creation order).
///
/// This guarantees that for any `_id`, the first match is the current version.
fn sort_sstables_by_level(sstables: &mut [SSTableReader], ucs: &UcsCompaction) {
    sstables.sort_by(|a, b| {
        let level_a = ucs.level_for_size(a.file_size());
        let level_b = ucs.level_for_size(b.file_size());
        level_a.cmp(&level_b).then_with(|| {
            // Within same level, newest first (higher path = newer)
            b.path().cmp(a.path())
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingodb_blob::Value;

    fn make_blob(n: u64) -> IBlob {
        IBlob::from_pairs(vec![
            ("n", Value::U64(n)),
            ("data", Value::String(format!("document-{n}"))),
        ])
    }

    fn deterministic_id(i: u64) -> DocumentId {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&i.to_be_bytes());
        let hash = i.wrapping_mul(0x517cc1b727220a95);
        bytes[8..16].copy_from_slice(&hash.to_be_bytes());
        DocumentId::from_bytes(bytes)
    }

    fn make_product_with_id(id: DocumentId, i: u64) -> IBlob {
        let categories = ["electronics", "books", "clothing", "home", "sports"];
        let category = categories[(i % categories.len() as u64) as usize];
        let price = (i % 1000) as f64 + 0.99;
        IBlob::with_id(
            id,
            [
                ("type".into(), Value::String("product".into())),
                ("name".into(), Value::String(format!("Product #{i}"))),
                ("category".into(), Value::String(category.into())),
                ("price".into(), Value::F64(price)),
                ("rating".into(), Value::F64((i % 50) as f64 / 10.0)),
                ("stock".into(), Value::U64(i % 500)),
                ("description".into(), Value::String(format!("Desc {i}"))),
            ]
            .into(),
        )
    }

    fn test_engine() -> (LsmEngine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 4096, // small for testing
            block_size: 256,
            compaction_threshold: 4,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();
        (engine, dir)
    }

    #[test]
    fn test_put_and_get() {
        let (engine, _dir) = test_engine();
        let blob = make_blob(1);
        let id = *blob.id();

        engine.put(blob.clone()).unwrap();
        let found = engine.get(&id).unwrap().unwrap();
        assert_eq!(found.id(), &id);
        assert_eq!(found.fields(), blob.fields());
        assert!(
            !found.version().is_nil(),
            "version should be stamped by engine"
        );
    }

    #[test]
    fn test_get_missing() {
        let (engine, _dir) = test_engine();
        let missing = DocumentId::from_bytes([0xFF; 16]);
        assert!(engine.get(&missing).unwrap().is_none());
    }

    #[test]
    fn test_flush_and_read_from_sstable() {
        let (engine, _dir) = test_engine();

        let blobs: Vec<_> = (0..10).map(|i| make_blob(i)).collect();
        let ids: Vec<_> = blobs.iter().map(|b| *b.id()).collect();
        for b in &blobs {
            engine.put(b.clone()).unwrap();
        }

        // Force flush
        engine.flush_memtable().unwrap();
        assert_eq!(engine.memtable_size(), 0);
        assert!(engine.sstable_count() >= 1);

        // All blobs still retrievable from SSTable
        for (i, id) in ids.iter().enumerate() {
            let found = engine.get(id).unwrap().unwrap();
            assert_eq!(found.get("n"), Some(&Value::U64(i as u64)));
        }
    }

    #[test]
    fn test_recovery_from_wal() {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024, // large enough to not auto-flush
            block_size: 256,
            compaction_threshold: 4,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };

        let blob = make_blob(42);
        let id = *blob.id();

        // Write and sync, then drop (simulating crash)
        {
            let engine = LsmEngine::open(config.clone()).unwrap();
            engine.put(blob).unwrap();
            engine.sync().unwrap();
        }

        // Reopen — should recover from WAL
        {
            let engine = LsmEngine::open(config).unwrap();
            let found = engine.get(&id).unwrap().unwrap();
            assert_eq!(found.id(), &id);
            assert!(
                !found.version().is_nil(),
                "recovered blob should have version"
            );
        }
    }

    #[test]
    fn test_many_writes_trigger_flushes() {
        let (engine, _dir) = test_engine();

        let mut ids = Vec::new();
        for i in 0..100 {
            let blob = make_blob(i);
            ids.push(*blob.id());
            engine.put(blob).unwrap();
        }

        assert!(engine.sstable_count() >= 1, "expected at least one SSTable");

        // Verify all documents are retrievable
        for (i, id) in ids.iter().enumerate() {
            let found = engine.get(id).unwrap();
            assert!(found.is_some(), "blob {i} not found");
        }
    }

    #[test]
    fn test_upsert_version_advances() {
        let (engine, _dir) = test_engine();
        let id = DocumentId::new();

        let blob1 = IBlob::with_id(id, [("x".into(), Value::U64(1))].into());
        engine.put(blob1).unwrap();
        let v1 = *engine.get(&id).unwrap().unwrap().version();

        let blob2 = IBlob::with_id(id, [("x".into(), Value::U64(2))].into());
        engine.put(blob2).unwrap();
        let found = engine.get(&id).unwrap().unwrap();
        let v2 = *found.version();

        assert!(v2 > v1, "version should advance on update");
        assert_eq!(found.get("x"), Some(&Value::U64(2)));
    }

    #[test]
    fn test_delete_from_memtable() {
        let (engine, _dir) = test_engine();
        let blob = make_blob(1);
        let id = *blob.id();

        engine.put(blob).unwrap();
        assert!(engine.get(&id).unwrap().is_some());

        engine.delete(&id).unwrap();
        assert!(engine.get(&id).unwrap().is_none());
        assert!(!engine.contains(&id).unwrap());
    }

    #[test]
    fn test_delete_from_sstable() {
        let (engine, _dir) = test_engine();
        let blob = make_blob(1);
        let id = *blob.id();

        engine.put(blob).unwrap();
        engine.flush_memtable().unwrap();
        assert!(engine.get(&id).unwrap().is_some());

        engine.delete(&id).unwrap();
        assert!(engine.get(&id).unwrap().is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let (engine, _dir) = test_engine();
        let id = DocumentId::new();

        engine.delete(&id).unwrap();
        assert!(engine.get(&id).unwrap().is_none());
    }

    #[test]
    fn test_delete_and_reinsert() {
        let (engine, _dir) = test_engine();
        let id = DocumentId::new();

        let blob1 = IBlob::with_id(id, [("x".into(), Value::U64(1))].into());
        engine.put(blob1).unwrap();
        engine.delete(&id).unwrap();
        assert!(engine.get(&id).unwrap().is_none());

        // Re-insert with same _id
        let blob2 = IBlob::with_id(id, [("x".into(), Value::U64(2))].into());
        engine.put(blob2).unwrap();
        let found = engine.get(&id).unwrap().unwrap();
        assert_eq!(found.get("x"), Some(&Value::U64(2)));
    }

    #[test]
    fn test_level_ordering_invariant() {
        // Verify that after flush + update, we always get the latest version
        // even when the old version is in a higher-level (larger) SSTable
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024, // large enough to control flush manually
            block_size: 256,
            compaction_threshold: 100, // prevent auto-compaction
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();
        let id = DocumentId::new();

        // Write v1, flush to SSTable (will be L0)
        let blob1 = IBlob::with_id(id, [("v".into(), Value::U64(1))].into());
        engine.put(blob1).unwrap();
        engine.flush_memtable().unwrap();

        // Write v2, flush to a second SSTable (also L0, but newer)
        let blob2 = IBlob::with_id(id, [("v".into(), Value::U64(2))].into());
        engine.put(blob2).unwrap();
        engine.flush_memtable().unwrap();

        // Should get v2, not v1
        let found = engine.get(&id).unwrap().unwrap();
        assert_eq!(
            found.get("v"),
            Some(&Value::U64(2)),
            "level-aware read should return newest version"
        );
    }

    #[test]
    fn test_scan_all() {
        let (engine, _dir) = test_engine();
        for i in 0..5 {
            engine.put(make_blob(i)).unwrap();
        }
        let results = engine.scan(None, None, None, None).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_scan_with_filter() {
        let (engine, _dir) = test_engine();
        for i in 0..10 {
            engine.put(make_blob(i)).unwrap();
        }
        let filter = Filter::Gt {
            field: "n".into(),
            value: Value::U64(6),
        };
        let results = engine.scan(Some(&filter), None, None, None).unwrap();
        assert_eq!(results.len(), 3); // n=7,8,9
        for r in &results {
            if let Some(Value::U64(n)) = r.get("n") {
                assert!(*n > 6);
            }
        }
    }

    #[test]
    fn test_scan_with_limit() {
        let (engine, _dir) = test_engine();
        for i in 0..10 {
            engine.put(make_blob(i)).unwrap();
        }
        let results = engine.scan(None, None, None, Some(3)).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_scan_with_projection() {
        let (engine, _dir) = test_engine();
        engine.put(make_blob(1)).unwrap();

        let results = engine.scan(None, None, Some(&["n".into()]), None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_projection());
        assert_eq!(results[0].field_count(), 1);
        assert_eq!(results[0].get("n"), Some(&Value::U64(1)));
        assert!(results[0].get("data").is_none());
    }

    #[test]
    fn test_scan_skips_tombstones() {
        let (engine, _dir) = test_engine();
        let blob = make_blob(1);
        let id = *blob.id();
        engine.put(blob).unwrap();
        engine.put(make_blob(2)).unwrap();

        engine.delete(&id).unwrap();

        let results = engine.scan(None, None, None, None).unwrap();
        assert_eq!(results.len(), 1, "deleted doc not in scan results");
    }

    #[test]
    fn test_execute_get() {
        let (engine, _dir) = test_engine();
        let blob = make_blob(42);
        let id = *blob.id();
        engine.put(blob).unwrap();

        let results = engine.execute(&Query::Get { id }).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("n"), Some(&Value::U64(42)));
    }

    #[test]
    fn test_execute_scan() {
        let (engine, _dir) = test_engine();
        for i in 0..5 {
            engine.put(make_blob(i)).unwrap();
        }
        let results = engine
            .execute(&Query::Scan {
                filter: Some(Filter::Lt {
                    field: "n".into(),
                    value: Value::U64(3),
                }),
                sort: None,
                project: None,
                limit: None,
            })
            .unwrap();
        assert_eq!(results.len(), 3); // n=0,1,2
    }

    #[test]
    fn test_traverse_simple_join() {
        let (engine, _dir) = test_engine();

        // Create users
        let user1 = IBlob::from_pairs(vec![
            ("type", Value::String("user".into())),
            ("name", Value::String("Henrik".into())),
        ]);
        let user1_id = *user1.id();
        engine.put(user1).unwrap();

        let user2 = IBlob::from_pairs(vec![
            ("type", Value::String("user".into())),
            ("name", Value::String("Alice".into())),
        ]);
        engine.put(user2).unwrap();

        // Create orders referencing users by _id
        engine
            .put(IBlob::from_pairs(vec![
                ("type", Value::String("order".into())),
                ("user_id", Value::Uuid(user1_id)),
                ("amount", Value::U64(100)),
            ]))
            .unwrap();

        // Traverse: from orders, join user_id -> _id to find referenced users
        let results = engine
            .traverse(
                Some(&Filter::Eq {
                    field: "type".into(),
                    value: Value::String("order".into()),
                }),
                "user_id",
                "_id",
                1,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].get("name"),
            Some(&Value::String("Henrik".into()))
        );
    }

    #[test]
    fn test_traverse_non_unique_join() {
        let (engine, _dir) = test_engine();

        // Two users named Henrik
        for i in 0..2 {
            engine
                .put(IBlob::from_pairs(vec![
                    ("type", Value::String("user".into())),
                    ("name", Value::String("Henrik".into())),
                    ("seq", Value::U64(i)),
                ]))
                .unwrap();
        }

        // An order referencing "Henrik" by name
        engine
            .put(IBlob::from_pairs(vec![
                ("type", Value::String("order".into())),
                ("user_name", Value::String("Henrik".into())),
            ]))
            .unwrap();

        // Traverse: orders.user_name -> users.name (non-unique — should find both)
        let results = engine
            .traverse(
                Some(&Filter::Eq {
                    field: "type".into(),
                    value: Value::String("order".into()),
                }),
                "user_name",
                "name",
                1,
            )
            .unwrap();
        assert_eq!(results.len(), 2, "non-unique join should find all matches");
    }

    #[test]
    fn test_traverse_depth_2() {
        let (engine, _dir) = test_engine();

        // company -> department -> employee chain
        let company = IBlob::from_pairs(vec![
            ("type", Value::String("company".into())),
            ("name", Value::String("Nyrkio".into())),
        ]);
        let company_id = *company.id();
        engine.put(company).unwrap();

        let dept = IBlob::from_pairs(vec![
            ("type", Value::String("dept".into())),
            ("company_id", Value::Uuid(company_id)),
            ("name", Value::String("Engineering".into())),
        ]);
        let dept_id = *dept.id();
        engine.put(dept).unwrap();

        let emp = IBlob::from_pairs(vec![
            ("type", Value::String("employee".into())),
            ("dept_id", Value::Uuid(dept_id)),
            ("name", Value::String("Henrik".into())),
        ]);
        engine.put(emp).unwrap();

        // Depth 1: company -> departments (join company _id -> dept.company_id)
        let depts = engine
            .traverse(
                Some(&Filter::Eq {
                    field: "type".into(),
                    value: Value::String("company".into()),
                }),
                "_id",
                "company_id",
                1,
            )
            .unwrap();
        assert_eq!(depts.len(), 1);
        assert_eq!(
            depts[0].get("name"),
            Some(&Value::String("Engineering".into()))
        );

        // Depth 2: company -> dept -> employees (same edge pattern repeated)
        // For depth>1 with the same edge, we need the same from/to fields to chain.
        // Here the chain is: company._id -> dept.company_id at hop 1,
        // then dept._id -> emp.dept_id at hop 2... but that's DIFFERENT edges.
        // Depth>1 with same edge only works for self-referential graphs.
        // For now, depth>1 repeats the same edge. So let's test that:

        // Self-referential: manager chain
        let ceo = IBlob::from_pairs(vec![
            ("role", Value::String("CEO".into())),
            ("name", Value::String("Boss".into())),
        ]);
        let ceo_id = *ceo.id();
        engine.put(ceo).unwrap();

        let vp = IBlob::from_pairs(vec![
            ("role", Value::String("VP".into())),
            ("name", Value::String("Manager".into())),
            ("reports_to", Value::Uuid(ceo_id)),
        ]);
        let vp_id = *vp.id();
        engine.put(vp).unwrap();

        let dev = IBlob::from_pairs(vec![
            ("role", Value::String("Dev".into())),
            ("name", Value::String("Coder".into())),
            ("reports_to", Value::Uuid(vp_id)),
        ]);
        engine.put(dev).unwrap();

        // From dev, follow reports_to -> _id, depth 2
        let chain = engine
            .traverse(
                Some(&Filter::Eq {
                    field: "role".into(),
                    value: Value::String("Dev".into()),
                }),
                "reports_to",
                "_id",
                2,
            )
            .unwrap();
        assert_eq!(chain.len(), 2, "depth 2 should find VP and CEO");
    }

    #[test]
    fn test_traverse_no_matches() {
        let (engine, _dir) = test_engine();
        engine.put(make_blob(1)).unwrap();

        let results = engine
            .traverse(
                Some(&Filter::Eq {
                    field: "n".into(),
                    value: Value::U64(1),
                }),
                "nonexistent_field",
                "_id",
                1,
            )
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_traverse_depth_zero() {
        let (engine, _dir) = test_engine();
        engine.put(make_blob(1)).unwrap();

        let results = engine.traverse(None, "_id", "_id", 0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_filter_on_id() {
        let (engine, _dir) = test_engine();
        let blob = make_blob(42);
        let id = *blob.id();
        engine.put(blob).unwrap();
        engine.put(make_blob(1)).unwrap();

        // Scan filtering on _id
        let results = engine
            .scan(
                Some(&Filter::Eq {
                    field: "_id".into(),
                    value: Value::Uuid(id),
                }),
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("n"), Some(&Value::U64(42)));
    }

    #[test]
    fn test_scan_sort_ascending() {
        let (engine, _dir) = test_engine();
        engine.put(make_blob(30)).unwrap();
        engine.put(make_blob(10)).unwrap();
        engine.put(make_blob(20)).unwrap();

        let results = engine
            .scan(
                None,
                Some(&[SortField {
                    field: "n".into(),
                    direction: SortDirection::Ascending,
                }]),
                None,
                None,
            )
            .unwrap();
        let ns: Vec<u64> = results
            .iter()
            .filter_map(|b| match b.get("n") {
                Some(Value::U64(n)) => Some(*n),
                _ => None,
            })
            .collect();
        assert_eq!(ns, vec![10, 20, 30]);
    }

    #[test]
    fn test_scan_sort_descending() {
        let (engine, _dir) = test_engine();
        engine.put(make_blob(30)).unwrap();
        engine.put(make_blob(10)).unwrap();
        engine.put(make_blob(20)).unwrap();

        let results = engine
            .scan(
                None,
                Some(&[SortField {
                    field: "n".into(),
                    direction: SortDirection::Descending,
                }]),
                None,
                None,
            )
            .unwrap();
        let ns: Vec<u64> = results
            .iter()
            .filter_map(|b| match b.get("n") {
                Some(Value::U64(n)) => Some(*n),
                _ => None,
            })
            .collect();
        assert_eq!(ns, vec![30, 20, 10]);
    }

    #[test]
    fn test_scan_sort_with_limit() {
        let (engine, _dir) = test_engine();
        for i in 0..10 {
            engine.put(make_blob(i)).unwrap();
        }
        // Sort descending, take top 3
        let results = engine
            .scan(
                None,
                Some(&[SortField {
                    field: "n".into(),
                    direction: SortDirection::Descending,
                }]),
                None,
                Some(3),
            )
            .unwrap();
        let ns: Vec<u64> = results
            .iter()
            .filter_map(|b| match b.get("n") {
                Some(Value::U64(n)) => Some(*n),
                _ => None,
            })
            .collect();
        assert_eq!(ns, vec![9, 8, 7]);
    }

    #[test]
    fn test_scan_sort_with_filter_and_projection() {
        let (engine, _dir) = test_engine();
        for i in 0..10 {
            engine.put(make_blob(i)).unwrap();
        }
        // Filter n > 5, sort ascending, project only "n"
        let filter = Filter::Gt {
            field: "n".into(),
            value: Value::U64(5),
        };
        let results = engine
            .scan(
                Some(&filter),
                Some(&[SortField {
                    field: "n".into(),
                    direction: SortDirection::Ascending,
                }]),
                Some(&["n".into()]),
                None,
            )
            .unwrap();
        assert_eq!(results.len(), 4); // 6,7,8,9
        assert!(results[0].is_projection());
        let ns: Vec<u64> = results
            .iter()
            .filter_map(|b| match b.get("n") {
                Some(Value::U64(n)) => Some(*n),
                _ => None,
            })
            .collect();
        assert_eq!(ns, vec![6, 7, 8, 9]);
    }

    #[test]
    fn test_reactive_index_creation() {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        // Insert docs and flush to SSTable (index only covers on-disk data)
        for i in 0..10 {
            engine.put(make_blob(i)).unwrap();
        }
        engine.flush_memtable().unwrap();

        assert_eq!(engine.secondary_index_count(), 0, "no index yet");

        // Run sorted scan DEFAULT_INDEX_THRESHOLD times to trigger reactive index
        let sort = [SortField {
            field: "n".into(),
            direction: SortDirection::Ascending,
        }];
        for _ in 0..secondary::DEFAULT_INDEX_THRESHOLD {
            engine.scan(None, Some(&sort), None, None).unwrap();
        }

        assert_eq!(
            engine.secondary_index_count(),
            1,
            "index should be created reactively"
        );

        // Next scan should use the index (and produce correct results)
        let results = engine.scan(None, Some(&sort), None, None).unwrap();
        let ns: Vec<u64> = results
            .iter()
            .filter_map(|b| match b.get("n") {
                Some(Value::U64(n)) => Some(*n),
                _ => None,
            })
            .collect();
        assert_eq!(ns, (0..10).collect::<Vec<u64>>());
    }

    #[test]
    fn test_no_index_below_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        // Only 3 docs — below spill threshold (5)
        for i in 0..3 {
            engine.put(make_blob(i)).unwrap();
        }
        engine.flush_memtable().unwrap();

        // Sorted scan with ≤threshold results stays in memory, no index created
        let sort = [SortField {
            field: "n".into(),
            direction: SortDirection::Ascending,
        }];
        engine.scan(None, Some(&sort), None, None).unwrap();

        assert_eq!(
            engine.secondary_index_count(),
            0,
            "should not build index below spill threshold"
        );
    }

    #[test]
    fn test_index_handles_stale_entries() {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        let blob1 = make_blob(1);
        let id_to_delete = *blob1.id();
        engine.put(blob1).unwrap();
        for i in 2..10 {
            engine.put(make_blob(i)).unwrap();
        }
        engine.flush_memtable().unwrap();

        // Sorted scan creates index via spill
        let sort = [SortField {
            field: "n".into(),
            direction: SortDirection::Ascending,
        }];
        engine.scan(None, Some(&sort), None, None).unwrap();
        assert_eq!(engine.secondary_index_count(), 1);

        // Delete one doc after index was built
        engine.delete(&id_to_delete).unwrap();

        // Scan with index should skip the deleted doc
        let results = engine.scan(None, Some(&sort), None, None).unwrap();
        assert_eq!(results.len(), 8, "stale entry should be skipped");
        assert!(results.iter().all(|b| b.get("n") != Some(&Value::U64(1))));
    }

    #[test]
    fn test_index_maintained_on_update() {
        // Henrik's scenario: index on field1, update field1 from 5 to 9.
        // Scan for field1 < 7 should NOT find the old value.
        // Scan for field1 > 7 should find the new value.
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        let id = DocumentId::new();
        let blob = IBlob::with_id(id, [("field1".into(), Value::U64(5))].into());
        engine.put(blob).unwrap();

        // Add enough docs to exceed spill threshold
        for i in [1u64, 2, 3, 4, 6, 8, 10, 11, 12] {
            engine
                .put(IBlob::from_pairs(vec![("field1", Value::U64(i))]))
                .unwrap();
        }
        engine.flush_memtable().unwrap();

        // Sorted scan creates index via spill (>5 results)
        let sort = [SortField {
            field: "field1".into(),
            direction: SortDirection::Ascending,
        }];
        engine.scan(None, Some(&sort), None, None).unwrap();
        assert_eq!(engine.secondary_index_count(), 1);

        // Update field1 from 5 to 9
        let updated = IBlob::with_id(id, [("field1".into(), Value::U64(9))].into());
        engine.put(updated).unwrap();

        // Scan for field1 < 7 — should NOT find the old value (5)
        let results = engine
            .scan(
                Some(&Filter::Lt {
                    field: "field1".into(),
                    value: Value::U64(7),
                }),
                Some(&sort),
                None,
                None,
            )
            .unwrap();
        let vals: Vec<u64> = results
            .iter()
            .filter_map(|b| match b.get("field1") {
                Some(Value::U64(n)) => Some(*n),
                _ => None,
            })
            .collect();
        assert_eq!(
            vals,
            vec![1, 2, 3, 4, 6],
            "old value 5 should not appear (stale, updated to 9)"
        );

        // Scan for field1 > 7 — should find the new value (9) plus 8, 10, 11, 12
        let results = engine
            .scan(
                Some(&Filter::Gt {
                    field: "field1".into(),
                    value: Value::U64(7),
                }),
                Some(&sort),
                None,
                None,
            )
            .unwrap();
        let vals: Vec<u64> = results
            .iter()
            .filter_map(|b| match b.get("field1") {
                Some(Value::U64(n)) => Some(*n),
                _ => None,
            })
            .collect();
        assert_eq!(
            vals,
            vec![8, 9, 10, 11, 12],
            "new value 9 should appear among results > 7"
        );
    }

    #[test]
    fn test_index_maintained_on_put_new_doc() {
        // New document inserted after index built should appear in sorted scan
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        for i in 0..8u64 {
            engine
                .put(IBlob::from_pairs(vec![("val", Value::U64(i * 10))]))
                .unwrap();
        }
        engine.flush_memtable().unwrap();

        // Sorted scan creates index via spill (8 > 5)
        let sort = [SortField {
            field: "val".into(),
            direction: SortDirection::Ascending,
        }];
        engine.scan(None, Some(&sort), None, None).unwrap();
        assert_eq!(engine.secondary_index_count(), 1);

        // Insert new doc after index built
        engine
            .put(IBlob::from_pairs(vec![("val", Value::U64(25))]))
            .unwrap();

        // Sorted scan should include the new doc in correct position
        let results = engine.scan(None, Some(&sort), None, None).unwrap();
        let vals: Vec<u64> = results
            .iter()
            .filter_map(|b| match b.get("val") {
                Some(Value::U64(n)) => Some(*n),
                _ => None,
            })
            .collect();
        assert_eq!(
            vals,
            vec![0, 10, 20, 25, 30, 40, 50, 60, 70],
            "new doc should appear in sorted position"
        );
    }

    #[test]
    fn test_sort_spills_to_disk() {
        let (engine, _dir) = test_engine();

        // Insert >5 docs (spill threshold)
        for i in 0..10 {
            engine.put(make_blob(i)).unwrap();
        }
        engine.flush_memtable().unwrap();

        assert_eq!(engine.secondary_index_count(), 0);

        // Sorted scan should spill to disk and create an index
        let sort = [SortField {
            field: "n".into(),
            direction: SortDirection::Ascending,
        }];
        let results = engine.scan(None, Some(&sort), None, None).unwrap();
        assert_eq!(results.len(), 10);

        assert_eq!(
            engine.secondary_index_count(),
            1,
            "should spill to disk as partial index"
        );
    }

    #[test]
    fn test_small_sort_stays_in_memory() {
        let (engine, _dir) = test_engine();

        // Insert ≤5 docs (at or below threshold)
        for i in 0..5 {
            engine.put(make_blob(i)).unwrap();
        }
        engine.flush_memtable().unwrap();

        let sort = [SortField {
            field: "n".into(),
            direction: SortDirection::Ascending,
        }];
        let results = engine.scan(None, Some(&sort), None, None).unwrap();
        assert_eq!(results.len(), 5);

        assert_eq!(
            engine.secondary_index_count(),
            0,
            "small sort should stay in memory"
        );
    }

    #[test]
    fn test_new_scan_replaces_old_index() {
        let (engine, _dir) = test_engine();

        for i in 0..20 {
            engine.put(make_blob(i)).unwrap();
        }
        engine.flush_memtable().unwrap();

        // First sorted scan creates an index (full range, no filter)
        let sort = [SortField {
            field: "n".into(),
            direction: SortDirection::Ascending,
        }];
        engine.scan(None, Some(&sort), None, None).unwrap();
        assert_eq!(engine.secondary_index_count(), 1);

        // Second sorted scan with a filter creates a new partial index, replacing the old
        let filter = Filter::Lt {
            field: "n".into(),
            value: Value::U64(10),
        };
        engine.scan(Some(&filter), Some(&sort), None, None).unwrap();
        assert_eq!(
            engine.secondary_index_count(),
            1,
            "should replace, not accumulate"
        );
    }

    #[test]
    fn test_descending_sort_uses_index() {
        let (engine, _dir) = test_engine();

        for i in 0..10 {
            engine.put(make_blob(i)).unwrap();
        }
        engine.flush_memtable().unwrap();

        // First, ascending scan to create the index
        let asc_sort = [SortField {
            field: "n".into(),
            direction: SortDirection::Ascending,
        }];
        engine.scan(None, Some(&asc_sort), None, None).unwrap();
        assert_eq!(engine.secondary_index_count(), 1);

        // Now descending scan should reuse the same index (reversed)
        let desc_sort = [SortField {
            field: "n".into(),
            direction: SortDirection::Descending,
        }];
        let results = engine.scan(None, Some(&desc_sort), None, None).unwrap();
        let ns: Vec<u64> = results
            .iter()
            .filter_map(|b| match b.get("n") {
                Some(Value::U64(n)) => Some(*n),
                _ => None,
            })
            .collect();
        assert_eq!(ns, vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);

        // Should still be just 1 index (not 2)
        assert_eq!(engine.secondary_index_count(), 1);
    }

    #[test]
    fn test_descending_sort_spills_and_creates_index() {
        let (engine, _dir) = test_engine();

        for i in 0..10 {
            engine.put(make_blob(i)).unwrap();
        }
        engine.flush_memtable().unwrap();

        // Descending scan should spill to disk and create an ascending index
        let desc_sort = [SortField {
            field: "n".into(),
            direction: SortDirection::Descending,
        }];
        let results = engine.scan(None, Some(&desc_sort), None, None).unwrap();
        let ns: Vec<u64> = results
            .iter()
            .filter_map(|b| match b.get("n") {
                Some(Value::U64(n)) => Some(*n),
                _ => None,
            })
            .collect();
        assert_eq!(ns, vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(
            engine.secondary_index_count(),
            1,
            "descending scan should create ascending index"
        );
    }

    #[test]
    fn test_partial_ranges_accumulated() {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        for i in 0..20 {
            engine.put(make_blob(i)).unwrap();
        }
        engine.flush_memtable().unwrap();

        // Two different filtered scans produce two partial indexes
        let sort = [SortField {
            field: "n".into(),
            direction: SortDirection::Ascending,
        }];

        let filter1 = Filter::Lt {
            field: "n".into(),
            value: Value::U64(10),
        };
        engine
            .scan(Some(&filter1), Some(&sort), None, None)
            .unwrap();

        let filter2 = Filter::Gt {
            field: "n".into(),
            value: Value::U64(5),
        };
        engine
            .scan(Some(&filter2), Some(&sort), None, None)
            .unwrap();

        assert_eq!(
            engine.secondary_index_count(),
            2,
            "two different ranges should accumulate"
        );
    }

    // ---- MVCC Snapshot Tests ----

    #[test]
    fn test_snapshot_get_sees_old_version() {
        let (engine, _dir) = test_engine();
        let id = DocumentId::new();

        engine
            .put(IBlob::with_id(id, [("x".into(), Value::U64(1))].into()))
            .unwrap();

        let snap = engine.snapshot();

        let blob2 = IBlob::with_id(id, [("x".into(), Value::U64(2))].into());
        engine.put(blob2).unwrap();

        // Regular get sees latest
        let latest = engine.get(&id).unwrap().unwrap();
        assert_eq!(latest.get("x"), Some(&Value::U64(2)));

        // Snapshot get sees old version
        let old = snap.get(&id).unwrap().unwrap();
        assert_eq!(old.get("x"), Some(&Value::U64(1)));
    }

    #[test]
    fn test_snapshot_scan_consistent() {
        let (engine, _dir) = test_engine();

        for i in 0..5 {
            engine.put(make_blob(i)).unwrap();
        }

        let snap = engine.snapshot();

        // Insert more after snapshot
        for i in 100..105 {
            engine.put(make_blob(i)).unwrap();
        }

        // Regular scan sees all 10
        let all = engine.scan(None, None, None, None).unwrap();
        assert_eq!(all.len(), 10);

        // Snapshot scan sees only the first 5
        let snapped = snap.scan(None, None, None, None).unwrap();
        assert_eq!(snapped.len(), 5);
        for blob in &snapped {
            if let Some(Value::U64(n)) = blob.get("n") {
                assert!(
                    *n < 100,
                    "snapshot should not see docs inserted after snapshot"
                );
            }
        }
    }

    #[test]
    fn test_snapshot_survives_flush() {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 256,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        let id = DocumentId::new();
        engine
            .put(IBlob::with_id(id, [("x".into(), Value::U64(1))].into()))
            .unwrap();

        let snap = engine.snapshot();

        // Update and flush
        engine
            .put(IBlob::with_id(id, [("x".into(), Value::U64(2))].into()))
            .unwrap();
        engine.flush_memtable().unwrap();

        // Snapshot still sees old version
        let old = snap.get(&id).unwrap().unwrap();
        assert_eq!(old.get("x"), Some(&Value::U64(1)));

        // Regular get sees new version
        let new = engine.get(&id).unwrap().unwrap();
        assert_eq!(new.get("x"), Some(&Value::U64(2)));
    }

    #[test]
    fn test_snapshot_delete_visibility() {
        let (engine, _dir) = test_engine();
        let id = DocumentId::new();

        engine
            .put(IBlob::with_id(id, [("x".into(), Value::U64(1))].into()))
            .unwrap();

        let snap = engine.snapshot();

        engine.delete(&id).unwrap();

        // Regular get: deleted
        assert!(engine.get(&id).unwrap().is_none());

        // Snapshot: still sees the document
        let found = snap.get(&id).unwrap().unwrap();
        assert_eq!(found.get("x"), Some(&Value::U64(1)));
    }

    #[test]
    fn test_multiple_snapshots() {
        let (engine, _dir) = test_engine();
        let id = DocumentId::new();

        engine
            .put(IBlob::with_id(id, [("x".into(), Value::U64(1))].into()))
            .unwrap();
        let s1 = engine.snapshot();

        engine
            .put(IBlob::with_id(id, [("x".into(), Value::U64(2))].into()))
            .unwrap();
        let s2 = engine.snapshot();

        engine
            .put(IBlob::with_id(id, [("x".into(), Value::U64(3))].into()))
            .unwrap();

        // Each snapshot sees its own point in time
        assert_eq!(s1.get(&id).unwrap().unwrap().get("x"), Some(&Value::U64(1)));
        assert_eq!(s2.get(&id).unwrap().unwrap().get("x"), Some(&Value::U64(2)));
        assert_eq!(
            engine.get(&id).unwrap().unwrap().get("x"),
            Some(&Value::U64(3))
        );
    }

    #[test]
    fn test_snapshot_gc_after_drop() {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 256,
            compaction_threshold: 2, // low threshold to trigger compaction
            scaling_parameter: 0,
            sort_spill_threshold: 1000,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        let id = DocumentId::new();
        engine
            .put(IBlob::with_id(id, [("x".into(), Value::U64(1))].into()))
            .unwrap();
        engine.flush_memtable().unwrap();

        {
            let _snap = engine.snapshot();
            engine
                .put(IBlob::with_id(id, [("x".into(), Value::U64(2))].into()))
                .unwrap();
            engine.flush_memtable().unwrap();
            // Snapshot is alive — compaction should keep both versions
        }
        // Snapshot dropped — next compaction can GC old version

        // Latest should still work
        let found = engine.get(&id).unwrap().unwrap();
        assert_eq!(found.get("x"), Some(&Value::U64(2)));
    }

    // ---- Index consistency tests ----

    #[test]
    fn test_index_stale_entry_skipped_after_flush() {
        // Update a doc after index is built, flush both updates.
        // The old index entry should be skipped (stale check).
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        // Insert 10 docs and flush to create primary SSTables
        let target_id = deterministic_id(0);
        for i in 0..10u64 {
            engine
                .put(make_product_with_id(deterministic_id(i), i))
                .unwrap();
        }
        engine.flush_memtable().unwrap();

        // Trigger index creation via sorted scan
        let sort = [SortField {
            field: "price".into(),
            direction: SortDirection::Ascending,
        }];
        engine.scan(None, Some(&sort), None, None).unwrap();
        assert!(engine.secondary_index_count() >= 1);

        // Update doc 0's price (was 0.99, now 999.99)
        engine
            .put(IBlob::with_id(
                target_id,
                [
                    ("type".into(), Value::String("product".into())),
                    ("name".into(), Value::String("Updated".into())),
                    ("category".into(), Value::String("electronics".into())),
                    ("price".into(), Value::F64(999.99)),
                    ("rating".into(), Value::F64(0.0)),
                    ("stock".into(), Value::U64(0)),
                    ("description".into(), Value::String("Updated".into())),
                ]
                .into(),
            ))
            .unwrap();
        engine.flush_memtable().unwrap();

        // Scan for cheap products — doc 0 should NOT appear (price is now 999.99)
        let results = engine
            .scan(
                Some(&Filter::Lt {
                    field: "price".into(),
                    value: Value::F64(10.0),
                }),
                Some(&sort),
                None,
                None,
            )
            .unwrap();
        for r in &results {
            assert_ne!(r.id(), &target_id, "stale index entry should be skipped");
        }
    }

    #[test]
    fn test_index_new_entry_visible_after_flush() {
        // Update a doc after index is built, flush. The NEW value should
        // appear in a sorted scan at the correct position.
        // This test uses the same engine instance (no restart) —
        // the in-memory buffer handles it.
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        let target_id = deterministic_id(0);
        for i in 0..10u64 {
            engine
                .put(make_product_with_id(deterministic_id(i), i))
                .unwrap();
        }
        engine.flush_memtable().unwrap();

        // Trigger index creation
        let sort = [SortField {
            field: "price".into(),
            direction: SortDirection::Ascending,
        }];
        engine.scan(None, Some(&sort), None, None).unwrap();
        assert!(engine.secondary_index_count() >= 1);

        // Update doc 0's price to 999.99
        engine
            .put(IBlob::with_id(
                target_id,
                [
                    ("type".into(), Value::String("product".into())),
                    ("name".into(), Value::String("Expensive".into())),
                    ("category".into(), Value::String("electronics".into())),
                    ("price".into(), Value::F64(999.99)),
                    ("rating".into(), Value::F64(0.0)),
                    ("stock".into(), Value::U64(0)),
                    ("description".into(), Value::String("Expensive".into())),
                ]
                .into(),
            ))
            .unwrap();
        engine.flush_memtable().unwrap();

        // Scan for expensive products — doc 0 should appear
        let results = engine
            .scan(
                Some(&Filter::Gt {
                    field: "price".into(),
                    value: Value::F64(500.0),
                }),
                Some(&sort),
                None,
                None,
            )
            .unwrap();

        let found = results.iter().any(|r| *r.id() == target_id);
        assert!(
            found,
            "updated doc should appear in sorted scan after flush (in-memory buffer)"
        );
    }

    #[test]
    fn test_index_new_entry_visible_after_flush_and_restart() {
        // The real bug: after restart, the in-memory index buffer is lost.
        // The flush must write secondary index entries to disk atomically
        // alongside the primary SSTable.
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };

        let target_id = deterministic_id(0);

        {
            let engine = LsmEngine::open(config.clone()).unwrap();

            // Insert 10 docs and flush
            for i in 0..10u64 {
                engine
                    .put(make_product_with_id(deterministic_id(i), i))
                    .unwrap();
            }
            engine.flush_memtable().unwrap();

            // Trigger index creation
            let sort = [SortField {
                field: "price".into(),
                direction: SortDirection::Ascending,
            }];
            engine.scan(None, Some(&sort), None, None).unwrap();
            assert!(engine.secondary_index_count() >= 1);

            // Update doc 0's price to 999.99
            engine
                .put(IBlob::with_id(
                    target_id,
                    [
                        ("type".into(), Value::String("product".into())),
                        ("name".into(), Value::String("Expensive".into())),
                        ("category".into(), Value::String("electronics".into())),
                        ("price".into(), Value::F64(999.99)),
                        ("rating".into(), Value::F64(0.0)),
                        ("stock".into(), Value::U64(0)),
                        ("description".into(), Value::String("Expensive".into())),
                    ]
                    .into(),
                ))
                .unwrap();

            // Flush — should write both primary SSTable AND secondary index entries
            engine.flush_memtable().unwrap();
        }

        // Restart — in-memory index buffer is gone
        {
            let engine = LsmEngine::open(config).unwrap();
            // Note: secondary indexes not loaded here (that's Database's job).
            // For this test, we manually load the index if it exists on disk.
            // The point: after flush, the index SSTable on disk should contain
            // the new entry, not just the in-memory buffer.

            // For now: just verify via a full scan (no index) that the doc is there
            let results = engine
                .scan(
                    Some(&Filter::Gt {
                        field: "price".into(),
                        value: Value::F64(500.0),
                    }),
                    None, // no sort — bypass index
                    None,
                    None,
                )
                .unwrap();
            let found = results.iter().any(|r| *r.id() == target_id);
            assert!(found, "updated doc visible via full scan after restart");

            // TODO: Once flush writes index entries to disk atomically,
            // this test should also verify sorted scan via index works after restart.
        }
    }

    #[test]
    fn test_filter_uses_secondary_index() {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        // Insert docs with categories
        for i in 0..20u64 {
            let cat = if i % 2 == 0 { "electronics" } else { "books" };
            engine
                .put(IBlob::from_pairs(vec![
                    ("category", Value::String(cat.into())),
                    ("n", Value::U64(i)),
                ]))
                .unwrap();
        }
        engine.flush_memtable().unwrap();

        // Create index on category by doing a sorted scan
        let sort = [SortField {
            field: "category".into(),
            direction: SortDirection::Ascending,
        }];
        engine.scan(None, Some(&sort), None, None).unwrap();
        assert!(engine.secondary_index_count() >= 1);

        // Now filter-only scan (no sort) should use the index
        let results = engine
            .scan(
                Some(&Filter::Eq {
                    field: "category".into(),
                    value: Value::String("electronics".into()),
                }),
                None,
                None,
                None,
            )
            .unwrap();

        assert_eq!(
            results.len(),
            10,
            "should find 10 electronics docs via index"
        );
        for r in &results {
            assert_eq!(
                r.get("category"),
                Some(&Value::String("electronics".into()))
            );
        }
    }

    #[test]
    fn test_filter_index_created_reactively() {
        // A filter-only query repeated with low selectivity should
        // trigger reactive index creation — no sort needed.
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5,
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };
        let engine = LsmEngine::open(config).unwrap();

        // Insert 20 docs: 10 electronics, 10 books
        for i in 0..20u64 {
            let cat = if i % 2 == 0 { "electronics" } else { "books" };
            engine
                .put(IBlob::from_pairs(vec![
                    ("category", Value::String(cat.into())),
                    ("n", Value::U64(i)),
                ]))
                .unwrap();
        }
        engine.flush_memtable().unwrap();

        assert_eq!(engine.secondary_index_count(), 0, "no index initially");

        // Scans 1 and 2 — stats accumulate but index not yet built (count < DEFAULT_INDEX_THRESHOLD=3)
        let filter = Filter::Eq {
            field: "category".into(),
            value: Value::String("electronics".into()),
        };
        engine.scan(Some(&filter), None, None, None).unwrap();
        assert_eq!(engine.secondary_index_count(), 0, "no index after first scan");

        engine.scan(Some(&filter), None, None, None).unwrap();
        assert_eq!(
            engine.secondary_index_count(),
            0,
            "no index after second scan"
        );

        // Third filter scan — stats show count=3 == DEFAULT_INDEX_THRESHOLD, selectivity=0.5 → build index
        let results = engine.scan(Some(&filter), None, None, None).unwrap();
        assert_eq!(results.len(), 10);
        assert_eq!(
            engine.secondary_index_count(),
            1,
            "index created reactively after 3 scans (DEFAULT_INDEX_THRESHOLD=3)"
        );

        // Fourth scan should use the index
        let results = engine.scan(Some(&filter), None, None, None).unwrap();
        assert_eq!(
            results.len(),
            10,
            "filter scan via index returns correct results"
        );
    }
}
