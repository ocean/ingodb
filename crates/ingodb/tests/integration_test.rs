use ingodb::{Database, DocumentId, Filter, IBlob, LsmConfig, LsmEngine, Query, Value};

fn make_user(name: &str, age: u64) -> IBlob {
    IBlob::from_pairs(vec![
        ("type", Value::String("user".into())),
        ("name", Value::String(name.into())),
        ("age", Value::U64(age)),
    ])
}

fn make_order(user_id: DocumentId, amount: u64) -> IBlob {
    IBlob::from_pairs(vec![
        ("type", Value::String("order".into())),
        ("user", Value::Uuid(user_id)),
        ("amount", Value::U64(amount)),
    ])
}

fn test_engine() -> (LsmEngine, tempfile::TempDir) {
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
    let engine = LsmEngine::open(config).unwrap();
    (engine, dir)
}

#[test]
fn test_document_lifecycle() {
    let (engine, _dir) = test_engine();

    // Insert a user
    let user = make_user("Henrik", 42);
    let user_id = *user.id();
    engine.put(user).unwrap();

    // Insert orders referencing the user by stable _id
    let order1 = make_order(user_id, 100);
    let order2 = make_order(user_id, 250);
    let order1_id = *order1.id();
    let order2_id = *order2.id();
    engine.put(order1).unwrap();
    engine.put(order2).unwrap();

    // Retrieve and verify
    let found_user = engine.get(&user_id).unwrap().unwrap();
    assert_eq!(
        found_user.get("name"),
        Some(&Value::String("Henrik".into()))
    );
    assert!(!found_user.version().is_nil(), "version should be stamped");

    let found_order = engine.get(&order1_id).unwrap().unwrap();
    assert_eq!(found_order.get("user"), Some(&Value::Uuid(user_id)));
    assert_eq!(found_order.get("amount"), Some(&Value::U64(100)));

    // Verify references work: follow order → user via stable _id
    let order2_doc = engine.get(&order2_id).unwrap().unwrap();
    if let Some(Value::Uuid(ref_id)) = order2_doc.get("user") {
        let referenced_user = engine.get(ref_id).unwrap().unwrap();
        assert_eq!(
            referenced_user.get("name"),
            Some(&Value::String("Henrik".into()))
        );
    } else {
        panic!("expected Ref in order.user field");
    }
}

#[test]
fn test_independent_documents() {
    let (engine, _dir) = test_engine();

    // Two documents with same content are independent (different _ids)
    let blob1 = make_user("Duplicate", 10);
    let blob2 = make_user("Duplicate", 10);
    let id1 = *blob1.id();
    let id2 = *blob2.id();
    assert_ne!(id1, id2);

    engine.put(blob1).unwrap();
    engine.put(blob2).unwrap();

    // Both are independently retrievable
    let found1 = engine.get(&id1).unwrap().unwrap();
    let found2 = engine.get(&id2).unwrap().unwrap();
    assert_eq!(found1.get("name"), Some(&Value::String("Duplicate".into())));
    assert_eq!(found2.get("name"), Some(&Value::String("Duplicate".into())));
}

#[test]
fn test_upsert() {
    let (engine, _dir) = test_engine();

    let id = DocumentId::new();
    let blob1 = IBlob::with_id(
        id,
        [
            ("name".into(), Value::String("Henrik".into())),
            ("age".into(), Value::U64(42)),
        ]
        .into(),
    );
    engine.put(blob1).unwrap();
    let v1 = *engine.get(&id).unwrap().unwrap().version();

    // Update: same _id, different content
    let blob2 = IBlob::with_id(
        id,
        [
            ("name".into(), Value::String("Henrik".into())),
            ("age".into(), Value::U64(43)),
        ]
        .into(),
    );
    engine.put(blob2).unwrap();

    let found = engine.get(&id).unwrap().unwrap();
    assert_eq!(found.get("age"), Some(&Value::U64(43)), "update applied");
    assert!(found.version() > &v1, "version advanced");
    assert_eq!(found.id(), &id, "_id unchanged");
}

#[test]
fn test_version_server_assigned() {
    let (engine, _dir) = test_engine();

    let blob = make_user("Test", 1);
    assert!(blob.version().is_nil(), "version nil before put");

    let id = *blob.id();
    engine.put(blob).unwrap();

    let found = engine.get(&id).unwrap().unwrap();
    assert!(!found.version().is_nil(), "version stamped after put");
}

#[test]
fn test_survive_restart() {
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

    let user = make_user("Persistent", 99);
    let id = *user.id();

    {
        let engine = LsmEngine::open(config.clone()).unwrap();
        engine.put(user).unwrap();
        engine.sync().unwrap();
    }

    {
        let engine = LsmEngine::open(config).unwrap();
        let found = engine.get(&id).unwrap().unwrap();
        assert_eq!(found.get("name"), Some(&Value::String("Persistent".into())));
        assert!(!found.version().is_nil(), "version survives restart");
    }
}

#[test]
fn test_flush_and_recover_from_sstable() {
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

    let mut ids = Vec::new();

    {
        let engine = LsmEngine::open(config.clone()).unwrap();
        for i in 0..20 {
            let blob = make_user(&format!("User{i}"), i);
            ids.push(*blob.id());
            engine.put(blob).unwrap();
        }
        engine.flush_memtable().unwrap();
    }

    {
        let engine = LsmEngine::open(config).unwrap();
        for id in &ids {
            assert!(
                engine.get(id).unwrap().is_some(),
                "document not found after restart"
            );
        }
    }
}

#[test]
fn test_zero_copy_field_extraction() {
    let mut blob = make_user("ZeroCopy", 77);
    let encoded = blob.encode();

    let name = IBlob::extract_field(&encoded, "name").unwrap();
    assert_eq!(name, Value::String("ZeroCopy".into()));

    let age = IBlob::extract_field(&encoded, "age").unwrap();
    assert_eq!(age, Value::U64(77));
}

#[test]
fn test_filter_evaluation() {
    let blob = make_user("FilterTest", 30);
    let get_field = |name: &str| blob.get(name).cloned();

    let filter = Filter::Gt {
        field: "age".into(),
        value: Value::U64(25),
    };
    assert!(filter.matches(&get_field));

    let compound = Filter::And(vec![
        Filter::Eq {
            field: "name".into(),
            value: Value::String("FilterTest".into()),
        },
        Filter::Lt {
            field: "age".into(),
            value: Value::U64(50),
        },
    ]);
    assert!(compound.matches(&get_field));
}

#[test]
fn test_large_documents() {
    let (engine, _dir) = test_engine();

    let keys: Vec<String> = (0..100).map(|i| format!("field_{i:04}")).collect();
    let pairs: Vec<(&str, Value)> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), Value::U64(i as u64)))
        .collect();
    let blob = IBlob::from_pairs(pairs);
    let id = *blob.id();

    engine.put(blob).unwrap();
    let found = engine.get(&id).unwrap().unwrap();
    assert_eq!(found.field_count(), 100);
}

#[test]
fn test_high_throughput_writes() {
    let (engine, _dir) = test_engine();

    let mut ids = Vec::with_capacity(1000);
    for i in 0..1000u64 {
        let blob = IBlob::from_pairs(vec![
            ("seq", Value::U64(i)),
            ("payload", Value::Bytes(vec![0xAB; 64])),
        ]);
        ids.push(*blob.id());
        engine.put(blob).unwrap();
    }

    for id in &ids {
        assert!(engine.get(id).unwrap().is_some());
    }

    assert!(
        engine.sstable_count() >= 1,
        "expected SSTables after 1000 writes"
    );
}

#[test]
fn test_delete_and_get() {
    let (engine, _dir) = test_engine();

    let user = make_user("ToDelete", 25);
    let id = *user.id();
    engine.put(user).unwrap();
    assert!(engine.get(&id).unwrap().is_some());

    engine.delete(&id).unwrap();
    assert!(engine.get(&id).unwrap().is_none());
    assert!(!engine.contains(&id).unwrap());
}

#[test]
fn test_delete_survives_flush() {
    let (engine, _dir) = test_engine();

    let user = make_user("FlushDelete", 30);
    let id = *user.id();
    engine.put(user).unwrap();
    engine.flush_memtable().unwrap();

    engine.delete(&id).unwrap();
    engine.flush_memtable().unwrap();

    // After both flushes, the tombstone in the newer SSTable should shadow the live entry
    assert!(engine.get(&id).unwrap().is_none());
}

#[test]
fn test_delete_survives_restart() {
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

    let user = make_user("RestartDelete", 40);
    let id = *user.id();

    {
        let engine = LsmEngine::open(config.clone()).unwrap();
        engine.put(user).unwrap();
        engine.delete(&id).unwrap();
        engine.sync().unwrap();
    }

    {
        let engine = LsmEngine::open(config).unwrap();
        assert!(
            engine.get(&id).unwrap().is_none(),
            "delete survives restart"
        );
    }
}

#[test]
fn test_delete_with_graph_ref_stable() {
    let (engine, _dir) = test_engine();

    // User and order referencing user by _id
    let user = make_user("Henrik", 42);
    let user_id = *user.id();
    engine.put(user).unwrap();

    let order = make_order(user_id, 100);
    let order_id = *order.id();
    engine.put(order).unwrap();

    // Delete user — order's Ref still points to the user_id
    engine.delete(&user_id).unwrap();
    assert!(engine.get(&user_id).unwrap().is_none());

    // Order still exists, reference is intact (it's a dangling ref now, but stable)
    let found_order = engine.get(&order_id).unwrap().unwrap();
    assert_eq!(found_order.get("user"), Some(&Value::Uuid(user_id)));

    // Re-insert user — order's ref resolves again
    let user2 = IBlob::with_id(
        user_id,
        [
            ("type".into(), Value::String("user".into())),
            ("name".into(), Value::String("Henrik v2".into())),
            ("age".into(), Value::U64(43)),
        ]
        .into(),
    );
    engine.put(user2).unwrap();
    let found_user = engine.get(&user_id).unwrap().unwrap();
    assert_eq!(
        found_user.get("name"),
        Some(&Value::String("Henrik v2".into()))
    );
}

#[test]
fn test_execute_get_query() {
    let (engine, _dir) = test_engine();
    let user = make_user("Henrik", 42);
    let id = *user.id();
    engine.put(user).unwrap();

    let results = engine.execute(&Query::Get { id }).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("name"),
        Some(&Value::String("Henrik".into()))
    );
}

#[test]
fn test_execute_scan_with_filter() {
    let (engine, _dir) = test_engine();
    engine.put(make_user("Alice", 25)).unwrap();
    engine.put(make_user("Bob", 35)).unwrap();
    engine.put(make_user("Charlie", 45)).unwrap();

    // Find users older than 30
    let results = engine
        .execute(&Query::Scan {
            filter: Some(Filter::Gt {
                field: "age".into(),
                value: Value::U64(30),
            }),
            sort: None,
            project: None,
            limit: None,
        })
        .unwrap();
    assert_eq!(results.len(), 2);
    for r in &results {
        if let Some(Value::U64(age)) = r.get("age") {
            assert!(*age > 30);
        }
    }
}

#[test]
fn test_execute_scan_with_projection() {
    let (engine, _dir) = test_engine();
    engine.put(make_user("Henrik", 42)).unwrap();
    engine.put(make_user("Alice", 30)).unwrap();

    let results = engine
        .execute(&Query::Scan {
            filter: None,
            sort: None,
            project: Some(vec!["name".into()]),
            limit: None,
        })
        .unwrap();
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.is_projection());
        assert_eq!(r.field_count(), 1);
        assert!(r.get("name").is_some());
        assert!(r.get("age").is_none(), "unprojected field should be absent");
        assert!(r.get("type").is_none());
    }
}

#[test]
fn test_execute_scan_with_limit() {
    let (engine, _dir) = test_engine();
    for i in 0..10 {
        engine.put(make_user(&format!("User{i}"), i)).unwrap();
    }
    let results = engine
        .execute(&Query::Scan {
            filter: None,
            sort: None,
            project: None,
            limit: Some(3),
        })
        .unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_scan_skips_deleted() {
    let (engine, _dir) = test_engine();
    let user1 = make_user("Keep", 20);
    let user2 = make_user("Delete", 30);
    let delete_id = *user2.id();
    engine.put(user1).unwrap();
    engine.put(user2).unwrap();
    engine.delete(&delete_id).unwrap();

    let results = engine.scan(None, None, None, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get("name"), Some(&Value::String("Keep".into())));
}

#[test]
fn test_traverse_orders_to_users() {
    let (engine, _dir) = test_engine();

    // Users
    let user1 = make_user("Henrik", 42);
    let user1_id = *user1.id();
    engine.put(user1).unwrap();

    let user2 = make_user("Alice", 30);
    let user2_id = *user2.id();
    engine.put(user2).unwrap();

    // Orders referencing users by _id stored as a Uuid field
    engine.put(make_order(user1_id, 100)).unwrap();
    engine.put(make_order(user1_id, 200)).unwrap();
    engine.put(make_order(user2_id, 50)).unwrap();

    // Find users referenced by orders over 75
    let results = engine
        .execute(&Query::Traverse {
            start: Some(Filter::Gt {
                field: "amount".into(),
                value: Value::U64(75),
            }),
            from_field: "user".into(),
            to_field: "_id".into(),
            depth: 1,
        })
        .unwrap();

    // Orders over 75: two orders for Henrik (100, 200). Should find Henrik (deduplicated).
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("name"),
        Some(&Value::String("Henrik".into()))
    );
}

#[test]
fn test_traverse_join_by_name() {
    let (engine, _dir) = test_engine();

    // Users and their reviews — joined by name, not by _id
    engine.put(make_user("Henrik", 42)).unwrap();
    engine.put(make_user("Alice", 30)).unwrap();

    engine
        .put(IBlob::from_pairs(vec![
            ("type", Value::String("review".into())),
            ("author", Value::String("Henrik".into())),
            ("text", Value::String("Great product".into())),
        ]))
        .unwrap();

    engine
        .put(IBlob::from_pairs(vec![
            ("type", Value::String("review".into())),
            ("author", Value::String("Alice".into())),
            ("text", Value::String("Not bad".into())),
        ]))
        .unwrap();

    // From Henrik's user doc, find reviews by joining name -> author
    let results = engine
        .execute(&Query::Traverse {
            start: Some(Filter::And(vec![
                Filter::Eq {
                    field: "type".into(),
                    value: Value::String("user".into()),
                },
                Filter::Eq {
                    field: "name".into(),
                    value: Value::String("Henrik".into()),
                },
            ])),
            from_field: "name".into(),
            to_field: "author".into(),
            depth: 1,
        })
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].get("text"),
        Some(&Value::String("Great product".into()))
    );
}

#[test]
fn test_query_stats_recorded() {
    let (engine, _dir) = test_engine();
    for i in 0..20 {
        engine.put(make_user(&format!("User{i}"), i)).unwrap();
    }

    // Run a filtered scan several times
    for _ in 0..5 {
        engine
            .execute(&Query::Scan {
                filter: Some(Filter::Gt {
                    field: "age".into(),
                    value: Value::U64(15),
                }),
                sort: None,
                project: None,
                limit: None,
            })
            .unwrap();
    }

    let stats = engine.query_stats().all_patterns();
    // Should have at least one "scan" pattern
    let scan_stats: Vec<_> = stats
        .iter()
        .filter(|(p, _)| p.query_type == "scan" && p.filter_fields.contains(&"age".into()))
        .collect();
    assert!(!scan_stats.is_empty(), "scan pattern should be recorded");

    let (_, ps) = &scan_stats[0];
    assert_eq!(ps.count, 5, "should record 5 executions");
    assert_eq!(
        ps.total_returned, 20,
        "4 docs returned × 5 runs (age 16,17,18,19)"
    );
    // A reactive filter index is built after scan 3 (DEFAULT_INDEX_THRESHOLD=3).
    // Scans 4-5 use the index and record docs_scanned=docs_returned=4 each.
    // Total: 20+20+20+4+4 = 68 — confirms the index reduced scan cost.
    assert_eq!(
        ps.total_scanned, 68,
        "first three scans full (20 each), then 2 via index (4 each)"
    );
}

#[test]
fn test_query_stats_get() {
    let (engine, _dir) = test_engine();
    let blob = make_user("Henrik", 42);
    let id = *blob.id();
    engine.put(blob).unwrap();

    engine.get(&id).unwrap();
    engine.get(&id).unwrap();
    engine.get(&DocumentId::new()).unwrap(); // miss

    let stats = engine.query_stats().all_patterns();
    let get_stats: Vec<_> = stats
        .iter()
        .filter(|(p, _)| p.query_type == "get")
        .collect();
    assert!(!get_stats.is_empty());

    let (_, ps) = &get_stats[0];
    assert_eq!(ps.count, 3, "3 get calls");
    assert_eq!(ps.total_returned, 2, "2 hits, 1 miss");
}

#[test]
fn test_query_stats_low_selectivity_detection() {
    let (engine, _dir) = test_engine();
    for i in 0..100 {
        engine
            .put(IBlob::from_pairs(vec![
                ("type", Value::String("item".into())),
                ("category", Value::String(format!("cat{}", i % 10))),
                ("seq", Value::U64(i)),
            ]))
            .unwrap();
    }

    // Query that scans 100 docs but returns only ~10 (category = "cat0")
    // After 2 scans with low selectivity, a reactive index is created.
    // Check stats after just 2 scans (before index changes the stats).
    for _ in 0..2 {
        engine
            .scan(
                Some(&Filter::Eq {
                    field: "category".into(),
                    value: Value::String("cat0".into()),
                }),
                None,
                None,
                None,
            )
            .unwrap();
    }

    // Detect index candidates — selectivity=0.1 (10 returned / 100 scanned)
    let candidates = engine.query_stats().low_selectivity(0.15, 2);
    assert!(
        !candidates.is_empty(),
        "should detect low-selectivity pattern"
    );
    assert!(candidates[0].0.filter_fields.contains(&"category".into()));
}

#[test]
fn test_filter_index_built_for_selective_queries() {
    // The reactive index trigger must be based on docs_scanned, not results.len().
    // A highly selective filter returns few docs but scans many — that is exactly
    // the case where an index gives the most benefit.
    //
    // With sort_spill_threshold=100 and 500 docs, 5 matching:
    //   OLD (wrong): results.len()=5 < 100 → no index ever built
    //   NEW (correct): docs_scanned=500 > 100 → index built after 2 scans
    let dir = tempfile::tempdir().unwrap();
    let config = LsmConfig {
        data_dir: dir.path().to_path_buf(),
        memtable_size: 1024 * 1024,
        block_size: 512,
        compaction_threshold: 100,
        scaling_parameter: 0,
        sort_spill_threshold: 100, // 5 results < 100 < 500 docs scanned
        compaction_threads: 1,
        adaptive_w: false,
        adaptive_w_cooldown_secs: 1,
        adaptive_w_max_step: 2,
        adaptive_w_min: -8,
        adaptive_w_max: 8,
    };
    let engine = LsmEngine::open(config).unwrap();

    // Insert 500 docs: only 5 have category='rare' (1% selectivity).
    for i in 0u64..500 {
        let category = if i < 5 { "rare" } else { "common" };
        engine
            .put(IBlob::from_pairs(vec![
                ("category", Value::String(category.into())),
                ("seq", Value::U64(i)),
            ]))
            .unwrap();
    }
    engine.flush_memtable().unwrap();

    let filter = Filter::Eq {
        field: "category".into(),
        value: Value::String("rare".into()),
    };

    assert_eq!(
        engine.secondary_index_count(),
        0,
        "no index before any scan"
    );

    // Scans 1 and 2: stats accumulate, index not yet built (count < DEFAULT_INDEX_THRESHOLD=3).
    let results1 = engine.scan(Some(&filter), None, None, None).unwrap();
    assert_eq!(results1.len(), 5, "first scan returns 5 docs");
    assert_eq!(
        engine.secondary_index_count(),
        0,
        "no index after first scan"
    );

    let results2 = engine.scan(Some(&filter), None, None, None).unwrap();
    assert_eq!(results2.len(), 5, "second scan returns 5 docs");
    assert_eq!(
        engine.secondary_index_count(),
        0,
        "no index after second scan"
    );

    // Third scan: stats.count=3 == DEFAULT_INDEX_THRESHOLD, docs_scanned=500 > threshold=100,
    // selectivity=0.01 < 0.5. Index is built reactively.
    let results3 = engine.scan(Some(&filter), None, None, None).unwrap();
    assert_eq!(results3.len(), 5, "third scan returns 5 docs");
    assert_eq!(
        engine.secondary_index_count(),
        1,
        "filter index built after third scan (DEFAULT_INDEX_THRESHOLD=3)"
    );

    // Fourth scan uses the index: O(log N + R) instead of O(N).
    let results4 = engine.scan(Some(&filter), None, None, None).unwrap();
    assert_eq!(results4.len(), 5, "fourth scan returns 5 docs via index");
    for doc in &results4 {
        assert_eq!(doc.get("category"), Some(&Value::String("rare".into())));
    }
}

#[test]
fn test_filter_index_not_built_below_scan_threshold() {
    // When docs_scanned <= sort_spill_threshold, no reactive index is built.
    // Avoids building indexes for cheap scans that don't need them.
    let dir = tempfile::tempdir().unwrap();
    let config = LsmConfig {
        data_dir: dir.path().to_path_buf(),
        memtable_size: 1024 * 1024,
        block_size: 512,
        compaction_threshold: 100,
        scaling_parameter: 0,
        sort_spill_threshold: 1000, // threshold higher than total docs
        compaction_threads: 1,
        adaptive_w: false,
        adaptive_w_cooldown_secs: 1,
        adaptive_w_max_step: 2,
        adaptive_w_min: -8,
        adaptive_w_max: 8,
    };
    let engine = LsmEngine::open(config).unwrap();

    for i in 0u64..100 {
        engine
            .put(IBlob::from_pairs(vec![
                ("category", Value::String("common".into())),
                ("seq", Value::U64(i)),
            ]))
            .unwrap();
    }
    engine.flush_memtable().unwrap();

    let filter = Filter::Eq {
        field: "category".into(),
        value: Value::String("common".into()),
    };

    // Run many scans — docs_scanned=100 < threshold=1000, so no index.
    for _ in 0..5 {
        engine.scan(Some(&filter), None, None, None).unwrap();
    }
    assert_eq!(
        engine.secondary_index_count(),
        0,
        "no index when scan cost is below threshold"
    );
}

#[test]
fn test_filter_index_survives_restart() {
    // A filter-field index persisted via the system collection must be loaded
    // on restart and used for subsequent queries.
    let dir = tempfile::tempdir().unwrap();
    let config = LsmConfig {
        data_dir: dir.path().to_path_buf(),
        memtable_size: 1024 * 1024,
        block_size: 512,
        compaction_threshold: 100,
        scaling_parameter: 0,
        sort_spill_threshold: 100,
        compaction_threads: 1,
        adaptive_w: false,
        adaptive_w_cooldown_secs: 1,
        adaptive_w_max_step: 2,
        adaptive_w_min: -8,
        adaptive_w_max: 8,
    };

    let target_ids: Vec<DocumentId>;

    {
        let db = Database::open(config.clone()).unwrap();

        let mut ids = Vec::new();
        db.with_collection("items", |engine: &LsmEngine| {
            for i in 0u64..500 {
                let category = if i < 5 { "rare" } else { "common" };
                let blob = IBlob::from_pairs(vec![
                    ("category", Value::String(category.into())),
                    ("seq", Value::U64(i)),
                ]);
                ids.push(*blob.id());
                engine.put(blob)?;
            }
            engine.flush_memtable()
        })
        .unwrap();

        target_ids = ids;

        let filter = Filter::Eq {
            field: "category".into(),
            value: Value::String("rare".into()),
        };

        // Three scans to trigger reactive index creation (DEFAULT_INDEX_THRESHOLD=3).
        for _ in 0..3 {
            db.with_collection("items", |engine: &LsmEngine| {
                engine.scan(Some(&filter), None, None, None)?;
                Ok(())
            })
            .unwrap();
        }

        let count = db
            .with_collection("items", |engine: &LsmEngine| {
                Ok(engine.secondary_index_count())
            })
            .unwrap();
        assert_eq!(count, 1, "filter index built before restart");
    }

    // Restart — index must be reloaded from system collection.
    {
        let db = Database::open(config).unwrap();

        let count = db
            .with_collection("items", |engine: &LsmEngine| {
                Ok(engine.secondary_index_count())
            })
            .unwrap();
        assert_eq!(count, 1, "filter index survives restart");

        let filter = Filter::Eq {
            field: "category".into(),
            value: Value::String("rare".into()),
        };
        let results = db
            .with_collection("items", |engine: &LsmEngine| {
                engine.scan(Some(&filter), None, None, None)
            })
            .unwrap();
        assert_eq!(
            results.len(),
            5,
            "correct results via reloaded filter index"
        );
    }

    let _ = target_ids;
}

/// Regression test: merging multiple partial indexes must perform a full rebuild
/// from primary SSTables, not just union the partial entries.
///
/// If partial entries were naively merged into a "full range" index, a subsequent
/// no-filter sort scan would miss documents that don't satisfy either partial filter.
#[test]
fn test_partial_index_merge_produces_full_coverage() {
    // Low spill threshold so partial indexes are created easily.
    let dir = tempfile::tempdir().unwrap();
    let config = LsmConfig {
        data_dir: dir.path().to_path_buf(),
        memtable_size: 4096,
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
    let engine = LsmEngine::open(config).unwrap();

    // Insert 30 docs: 10 "alpha", 10 "beta", 10 "gamma".
    for i in 0..10u64 {
        engine
            .put(IBlob::from_pairs(vec![
                ("category", Value::String("alpha".into())),
                ("score", Value::U64(i)),
            ]))
            .unwrap();
    }
    for i in 0..10u64 {
        engine
            .put(IBlob::from_pairs(vec![
                ("category", Value::String("beta".into())),
                ("score", Value::U64(i + 10)),
            ]))
            .unwrap();
    }
    for i in 0..10u64 {
        engine
            .put(IBlob::from_pairs(vec![
                ("category", Value::String("gamma".into())),
                ("score", Value::U64(i + 20)),
            ]))
            .unwrap();
    }

    use ingodb::{SortDirection, SortField};
    let sort = vec![SortField {
        field: "score".into(),
        direction: SortDirection::Ascending,
    }];

    // Two filtered sort scans → one partial index each (alpha range, beta range).
    let alpha_filter = Filter::Eq {
        field: "category".into(),
        value: Value::String("alpha".into()),
    };
    let r1 = engine
        .scan(Some(&alpha_filter), Some(&sort), None, None)
        .unwrap();
    assert_eq!(r1.len(), 10, "alpha scan returns 10 docs");

    let beta_filter = Filter::Eq {
        field: "category".into(),
        value: Value::String("beta".into()),
    };
    let r2 = engine
        .scan(Some(&beta_filter), Some(&sort), None, None)
        .unwrap();
    assert_eq!(r2.len(), 10, "beta scan returns 10 docs");

    // Compaction merges the two partial indexes via full rebuild.
    engine.compact_now().unwrap();

    // A no-filter sort scan must return ALL 30 docs.
    // The old buggy merge (union of partial entries) would return only 20.
    let all = engine.scan(None, Some(&sort), None, None).unwrap();
    assert_eq!(
        all.len(),
        30,
        "full sort scan returns all docs after partial index merge"
    );

    // Verify ascending sort order.
    let scores: Vec<u64> = all
        .iter()
        .map(|b| match b.get_field("score") {
            Some(Value::U64(v)) => v,
            _ => panic!("missing score field"),
        })
        .collect();
    assert_eq!(
        scores,
        (0u64..30).collect::<Vec<_>>(),
        "docs in ascending score order"
    );
}
