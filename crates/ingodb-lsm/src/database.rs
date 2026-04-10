use crate::{LsmConfig, LsmEngine, LsmError};
use ingodb_blob::{IBlob, Value};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A database containing multiple named collections.
///
/// Each collection is an independent LsmEngine with its own subdirectory,
/// WAL, memtable, and SSTables. The `system` collection is reserved for
/// engine metadata (e.g., index definitions) and always exists.
pub struct Database {
    data_dir: PathBuf,
    config: LsmConfig,
    collections: Mutex<HashMap<String, LsmEngine>>,
}

impl Database {
    /// Open or create a database at the given directory.
    /// Creates the `system` collection if it doesn't exist.
    /// Discovers and opens existing collection subdirectories.
    /// Loads secondary index metadata from the system collection.
    pub fn open(config: LsmConfig) -> Result<Self, LsmError> {
        std::fs::create_dir_all(&config.data_dir)?;

        let data_dir = config.data_dir.clone();
        let mut collections = HashMap::new();

        // Discover existing collection directories
        for entry in std::fs::read_dir(&data_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    let coll_config = Self::collection_config(&config, &data_dir, name);
                    match LsmEngine::open(coll_config) {
                        Ok(engine) => {
                            collections.insert(name.to_string(), engine);
                        }
                        Err(e) => {
                            eprintln!("warning: failed to open collection {name}: {e}");
                        }
                    }
                }
            }
        }

        // Ensure system collection exists
        if !collections.contains_key("system") {
            let sys_config = Self::collection_config(&config, &data_dir, "system");
            let engine = LsmEngine::open(sys_config)?;
            collections.insert("system".to_string(), engine);
        }

        // Load secondary index metadata from system collection
        let system = collections.get("system").unwrap();
        let all_docs = system.scan(None, None, None, None)?;
        for doc in &all_docs {
            if doc.get("type") == Some(&Value::String("index".into())) {
                let coll_name = match doc.get("collection") {
                    Some(Value::String(s)) => s.clone(),
                    _ => continue,
                };
                let fields_str = match doc.get("fields") {
                    Some(Value::String(s)) => s.clone(),
                    _ => continue,
                };
                let idx_path = match doc.get("path") {
                    Some(Value::String(s)) => PathBuf::from(s),
                    _ => continue,
                };

                let fields: Vec<String> = fields_str.split(',').map(|s| s.to_string()).collect();

                if let Some(engine) = collections.get(&coll_name) {
                    if idx_path.exists() {
                        if let Err(e) = engine.load_secondary_index(fields, None, &idx_path) {
                            eprintln!("warning: failed to load index {}: {e}", idx_path.display());
                        }
                    }
                }
            }
        }

        // Replay WAL-recovered memtable entries into secondary indexes
        for engine in collections.values() {
            engine.replay_memtable_to_indexes();
        }

        Ok(Database {
            data_dir,
            config,
            collections: Mutex::new(collections),
        })
    }

    /// Ensure a collection exists.
    pub fn collection(&self, name: &str) -> Result<(), LsmError> {
        let mut collections = self.collections.lock();
        if !collections.contains_key(name) {
            let coll_config = Self::collection_config(&self.config, &self.data_dir, name);
            let engine = LsmEngine::open(coll_config)?;
            collections.insert(name.to_string(), engine);
        }
        Ok(())
    }

    /// Execute a closure with access to a collection's engine.
    /// After the closure, persists any newly built index metadata to the system collection.
    pub fn with_collection<F, R>(&self, name: &str, f: F) -> Result<R, LsmError>
    where
        F: FnOnce(&LsmEngine) -> Result<R, LsmError>,
    {
        self.collection(name)?;
        let collections = self.collections.lock();
        let engine = collections.get(name).unwrap();
        let result = f(engine)?;

        // Persist any newly built secondary indexes
        let pending = engine.drain_pending_index_metadata();
        if !pending.is_empty() {
            if let Some(system) = collections.get("system") {
                for meta in pending {
                    let fields_str = meta.fields.join(",");
                    let path_str = meta.path.to_string_lossy().to_string();
                    let blob = IBlob::from_pairs(vec![
                        ("type", Value::String("index".into())),
                        ("collection", Value::String(name.to_string())),
                        ("fields", Value::String(fields_str)),
                        ("path", Value::String(path_str)),
                    ]);
                    system.put(blob)?;
                }
            }
        }

        Ok(result)
    }

    /// Access the system collection directly.
    pub fn system<F, R>(&self, f: F) -> Result<R, LsmError>
    where
        F: FnOnce(&LsmEngine) -> Result<R, LsmError>,
    {
        self.with_collection("system", f)
    }

    /// List all collection names.
    pub fn list_collections(&self) -> Vec<String> {
        let collections = self.collections.lock();
        let mut names: Vec<String> = collections.keys().cloned().collect();
        names.sort();
        names
    }

    /// Drop a collection. Cannot drop the `system` collection.
    pub fn drop_collection(&self, name: &str) -> Result<(), LsmError> {
        if name == "system" {
            return Err(LsmError::NotImplemented(
                "cannot drop system collection".into(),
            ));
        }
        let mut collections = self.collections.lock();
        collections.remove(name);
        let coll_dir = self.data_dir.join(name);
        if coll_dir.exists() {
            std::fs::remove_dir_all(&coll_dir)?;
        }
        Ok(())
    }

    /// Path to the database directory.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    fn collection_config(base: &LsmConfig, data_dir: &Path, name: &str) -> LsmConfig {
        let mut c = base.clone();
        c.data_dir = data_dir.join(name);
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secondary;
    use ingodb_blob::{DocumentId, Value};
    use ingodb_query::{Filter, SortDirection, SortField};

    fn test_db() -> (Database, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 8192,
            block_size: 512,
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
        let db = Database::open(config).unwrap();
        (db, dir)
    }

    #[test]
    fn test_system_collection_exists() {
        let (db, _dir) = test_db();
        let names = db.list_collections();
        assert!(names.contains(&"system".to_string()));
    }

    #[test]
    fn test_create_and_list_collections() {
        let (db, _dir) = test_db();

        db.collection("users").unwrap();
        db.collection("orders").unwrap();

        let names = db.list_collections();
        assert!(names.contains(&"users".to_string()));
        assert!(names.contains(&"orders".to_string()));
        assert!(names.contains(&"system".to_string()));
    }

    #[test]
    fn test_collections_isolated() {
        let (db, _dir) = test_db();

        let blob = IBlob::from_pairs(vec![("name", Value::String("Henrik".into()))]);
        let id = *blob.id();

        db.with_collection("users", |engine| {
            engine.put(blob)?;
            Ok(())
        })
        .unwrap();

        let found = db
            .with_collection("users", |engine| engine.get(&id))
            .unwrap();
        assert!(found.is_some());

        let found = db
            .with_collection("orders", |engine| engine.get(&id))
            .unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_system_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
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

        let blob = IBlob::from_pairs(vec![
            ("type", Value::String("index".into())),
            ("collection", Value::String("users".into())),
        ]);
        let id = *blob.id();

        {
            let db = Database::open(config.clone()).unwrap();
            db.system(|engine| {
                engine.put(blob)?;
                engine.sync()?;
                Ok(())
            })
            .unwrap();
        }

        {
            let db = Database::open(config).unwrap();
            let found = db.system(|engine| engine.get(&id)).unwrap();
            assert!(found.is_some(), "system data survives restart");
        }
    }

    #[test]
    fn test_drop_collection() {
        let (db, _dir) = test_db();

        db.collection("temp").unwrap();
        assert!(db.list_collections().contains(&"temp".to_string()));

        db.drop_collection("temp").unwrap();
        assert!(!db.list_collections().contains(&"temp".to_string()));
    }

    #[test]
    fn test_cannot_drop_system() {
        let (db, _dir) = test_db();
        assert!(db.drop_collection("system").is_err());
    }

    #[test]
    fn test_collections_survive_restart() {
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
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

        let blob = IBlob::from_pairs(vec![("x", Value::U64(42))]);
        let id = *blob.id();

        {
            let db = Database::open(config.clone()).unwrap();
            db.with_collection("mydata", |engine| {
                engine.put(blob)?;
                engine.flush_memtable()?;
                Ok(())
            })
            .unwrap();
        }

        {
            let db = Database::open(config).unwrap();
            assert!(db.list_collections().contains(&"mydata".to_string()));
            let found = db
                .with_collection("mydata", |engine| engine.get(&id))
                .unwrap();
            assert!(found.is_some(), "collection data survives restart");
        }
    }

    #[test]
    fn test_index_metadata_persisted_in_system() {
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

        {
            let db = Database::open(config.clone()).unwrap();

            // Insert docs and flush
            db.with_collection("users", |engine| {
                for i in 0..10u64 {
                    engine.put(IBlob::from_pairs(vec![
                        ("name", Value::String(format!("User{i}"))),
                        ("age", Value::U64(i)),
                    ]))?;
                }
                engine.flush_memtable()?;
                Ok(())
            })
            .unwrap();

            // Trigger reactive index creation
            let sort = [SortField {
                field: "age".into(),
                direction: SortDirection::Ascending,
            }];
            for _ in 0..secondary::DEFAULT_INDEX_THRESHOLD {
                db.with_collection("users", |engine| {
                    engine.scan(None, Some(&sort), None, None)?;
                    Ok(())
                })
                .unwrap();
            }

            // Verify index was built
            let count = db
                .with_collection("users", |engine| Ok(engine.secondary_index_count()))
                .unwrap();
            assert_eq!(count, 1, "index should be built");

            // Verify metadata in system collection
            let system_docs = db
                .system(|engine| engine.scan(None, None, None, None))
                .unwrap();
            let index_docs: Vec<_> = system_docs
                .iter()
                .filter(|d| d.get("type") == Some(&Value::String("index".into())))
                .collect();
            assert_eq!(index_docs.len(), 1, "index metadata should be in system");
            assert_eq!(
                index_docs[0].get("collection"),
                Some(&Value::String("users".into()))
            );
            assert_eq!(
                index_docs[0].get("fields"),
                Some(&Value::String("age".into()))
            );
        }

        // Restart and verify index is loaded
        {
            let db = Database::open(config).unwrap();
            let count = db
                .with_collection("users", |engine| Ok(engine.secondary_index_count()))
                .unwrap();
            assert_eq!(
                count, 1,
                "index should survive restart via system collection"
            );

            // Verify it works
            let sort = [SortField {
                field: "age".into(),
                direction: SortDirection::Ascending,
            }];
            let results = db
                .with_collection("users", |engine| engine.scan(None, Some(&sort), None, None))
                .unwrap();
            assert_eq!(results.len(), 10);
        }
    }

    #[test]
    fn test_index_buffer_survives_restart() {
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

        let update_id;

        {
            let db = Database::open(config.clone()).unwrap();

            // Insert docs and flush to create primary SSTables
            db.with_collection("data", |engine| {
                for i in 0..10u64 {
                    engine.put(IBlob::from_pairs(vec![("val", Value::U64(i))]))?;
                }
                engine.flush_memtable()?;
                Ok(())
            })
            .unwrap();

            // Trigger index creation via sorted scan
            let sort = [SortField {
                field: "val".into(),
                direction: SortDirection::Ascending,
            }];
            db.with_collection("data", |engine| {
                engine.scan(None, Some(&sort), None, None)?;
                Ok(())
            })
            .unwrap();

            // Now update a doc (this writes to index buffer)
            update_id = DocumentId::new();
            db.with_collection("data", |engine| {
                engine.put(IBlob::with_id(
                    update_id,
                    [("val".into(), Value::U64(99))].into(),
                ))?;
                engine.sync()?;
                Ok(())
            })
            .unwrap();

            // Don't flush — the update is only in WAL + memtable + index buffer
        }

        // Restart — WAL recovery should replay into index buffers
        {
            let db = Database::open(config).unwrap();

            // The updated doc should appear in sorted scan via the index
            let sort = [SortField {
                field: "val".into(),
                direction: SortDirection::Ascending,
            }];
            let results = db
                .with_collection("data", |engine| engine.scan(None, Some(&sort), None, None))
                .unwrap();

            let vals: Vec<u64> = results
                .iter()
                .filter_map(|b| match b.get("val") {
                    Some(Value::U64(n)) => Some(*n),
                    _ => None,
                })
                .collect();
            assert!(
                vals.contains(&99),
                "WAL-recovered doc should be in index after restart"
            );
        }
    }

    #[test]
    fn test_index_entry_survives_flush_and_restart() {
        // BUG TEST: After restart, a sorted scan through a secondary index
        // must find docs that were updated+flushed before the crash.
        // The index on disk must contain entries for the flushed data.
        //
        // To truly test the index path (not full-scan fallback), we set
        // sort_spill_threshold very high so the post-restart scan doesn't
        // create a NEW index from a full scan. This forces it to use the
        // existing index loaded from disk.
        let dir = tempfile::tempdir().unwrap();
        let config = LsmConfig {
            data_dir: dir.path().to_path_buf(),
            memtable_size: 1024 * 1024,
            block_size: 512,
            compaction_threshold: 100,
            scaling_parameter: 0,
            sort_spill_threshold: 5, // low: creates index on first scan
            compaction_threads: 1,
            adaptive_w: false,
            adaptive_w_cooldown_secs: 1,
            adaptive_w_max_step: 2,
            adaptive_w_min: -8,
            adaptive_w_max: 8,
        };

        fn det_id(i: u64) -> DocumentId {
            let mut bytes = [0u8; 16];
            bytes[..8].copy_from_slice(&i.to_be_bytes());
            let hash = i.wrapping_mul(0x517cc1b727220a95);
            bytes[8..16].copy_from_slice(&hash.to_be_bytes());
            DocumentId::from_bytes(bytes)
        }

        let target_id = det_id(0);

        {
            let db = Database::open(config.clone()).unwrap();

            // Insert 10 docs, flush
            db.with_collection("products", |engine| {
                for i in 0..10u64 {
                    engine.put(IBlob::with_id(
                        det_id(i),
                        [
                            ("price".into(), Value::F64((i % 100) as f64 + 0.99)),
                            ("name".into(), Value::String(format!("Product {i}"))),
                        ]
                        .into(),
                    ))?;
                }
                engine.flush_memtable()?;
                Ok(())
            })
            .unwrap();

            // Trigger index creation via sorted scan
            let sort = [SortField {
                field: "price".into(),
                direction: SortDirection::Ascending,
            }];
            db.with_collection("products", |engine| {
                engine.scan(None, Some(&sort), None, None)?;
                Ok(())
            })
            .unwrap();

            // Update doc 0: price 0.99 → 999.99
            db.with_collection("products", |engine| {
                engine.put(IBlob::with_id(
                    target_id,
                    [
                        ("price".into(), Value::F64(999.99)),
                        ("name".into(), Value::String("Expensive".into())),
                    ]
                    .into(),
                ))?;
                // Flush — must write secondary index entries atomically
                engine.flush_memtable()?;
                Ok(())
            })
            .unwrap();
        }

        // Restart — but with HIGH spill threshold so the scan doesn't create
        // a new index from full scan results. Forces use of the existing index.
        let config_high_spill = LsmConfig {
            sort_spill_threshold: 1_000_000, // won't spill
            ..config
        };

        {
            let db = Database::open(config_high_spill).unwrap();

            // Verify index exists after restart
            let idx_count = db
                .with_collection("products", |engine| Ok(engine.secondary_index_count()))
                .unwrap();
            assert!(idx_count >= 1, "index should exist after restart");

            // Sorted scan — should use the existing index from disk
            let sort = [SortField {
                field: "price".into(),
                direction: SortDirection::Ascending,
            }];
            let results = db
                .with_collection("products", |engine| {
                    engine.scan(None, Some(&sort), None, None)
                })
                .unwrap();

            // ALL 10 docs should be present
            assert_eq!(
                results.len(),
                10,
                "all 10 docs should be in sorted scan (got {})",
                results.len()
            );

            // The updated doc must be found with the new price
            let target = results.iter().find(|r| *r.id() == target_id);
            assert!(
                target.is_some(),
                "updated doc must appear in sorted scan via index after restart"
            );
            assert_eq!(
                target.unwrap().get("price"),
                Some(&Value::F64(999.99)),
                "updated doc should have new price after restart"
            );
        }
    }
}
