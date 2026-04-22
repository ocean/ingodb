//! E-commerce product catalog benchmark for IngoDB.
//!
//! Run with: cargo run --release --example benchmark
//!
//! Simulates a realistic workload:
//! 1. Bulk ingest of product documents
//! 2. Point lookups (get by _id)
//! 3. Scan + sort queries (with reactive index creation)
//! 4. Snapshot reads under concurrent writes
//! 5. Compaction performance

use ingodb::{
    DocumentId, Filter, IBlob, LsmConfig, LsmEngine, Query, SortDirection, SortField, Value,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const NUM_PRODUCTS: u64 = 1_000_000;
const NUM_LOOKUPS: u64 = 20_000;
const CATEGORIES: &[&str] = &[
    "electronics",
    "books",
    "clothing",
    "home",
    "sports",
    "toys",
    "food",
    "automotive",
    "garden",
    "health",
];

fn main() {
    let w: i32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Use ./data/ for real disk I/O (not tmpfs)
    let data_dir = PathBuf::from("data");
    if data_dir.exists() {
        std::fs::remove_dir_all(&data_dir).unwrap();
    }
    std::fs::create_dir_all(&data_dir).unwrap();

    let config = LsmConfig {
        data_dir: data_dir.clone(),
        memtable_size: 16 * 1024 * 1024, // 16 MB
        block_size: 4096,
        compaction_threshold: 4,
        scaling_parameter: w,
        sort_spill_threshold: 1000,
        compaction_threads: 4,
        adaptive_w: w == 0, // enable adaptive W when starting from balanced
        adaptive_w_cooldown_secs: 1, // very short cooldown for benchmarks
        adaptive_w_max_step: 16, // effectively unlimited — jump to target immediately
        adaptive_w_min: -8,
        adaptive_w_max: 8,
    };

    let mode = match w.cmp(&0) {
        std::cmp::Ordering::Less => format!("leveled (W={w}, f={}, t=2)", 2 - w),
        std::cmp::Ordering::Equal => "balanced (W=0, f=2, t=2)".into(),
        std::cmp::Ordering::Greater => format!("tiered (W={w}, f={}, t={})", 2 + w, 2 + w),
    };

    println!("=== IngoDB Benchmark: E-commerce Product Catalog ===\n");
    println!(
        "Config: {} products, {} MB memtable, {} byte blocks, UCS {}\n",
        NUM_PRODUCTS,
        config.memtable_size / 1024 / 1024,
        config.block_size,
        mode,
    );

    let engine = Arc::new(LsmEngine::open(config).unwrap());
    engine.start_background_compaction();

    let report_w = |label: &str, engine: &LsmEngine| {
        println!(
            "  [{label}] W: effective={}, target={}, SSTables={}",
            engine.effective_w(),
            engine.target_w(),
            engine.sstable_count()
        );
    };

    // Phase 1: Bulk ingest (write-heavy)
    let ids = phase_bulk_ingest(&engine);
    let t = Instant::now();
    engine.wait_for_compaction().unwrap();
    println!("  Compaction settled in {:?}", t.elapsed());
    report_w("after ingest", &engine);

    // Phase 1b: Random updates — 2x the docs (write-heavy)
    phase_random_updates(&engine, &ids);
    let t = Instant::now();
    engine.wait_for_compaction().unwrap();
    println!("  Compaction settled in {:?}", t.elapsed());
    report_w("after updates", &engine);

    // Phase 2: Point lookups — 2x (read-heavy)
    phase_point_lookups(&engine, &ids);
    let t = Instant::now();
    engine.wait_for_compaction().unwrap();
    println!("  Compaction settled in {:?}", t.elapsed());
    report_w("after lookups", &engine);

    // Phase 3: Scan + sort queries (read-heavy)
    phase_scan_queries(&engine);
    let t = Instant::now();
    engine.wait_for_compaction().unwrap();
    println!("  Compaction settled in {:?}", t.elapsed());
    report_w("after scans", &engine);

    // Phase 4: Snapshot reads
    phase_snapshot_reads(&engine, &ids);

    // Phase 5: Mixed read/write
    phase_mixed_workload(&engine, &ids);
    let t = Instant::now();
    engine.wait_for_compaction().unwrap();
    println!("  Compaction settled in {:?}", t.elapsed());
    report_w("after mixed", &engine);

    // Phase 6: Concurrent reads
    phase_concurrent_reads(&engine, &ids);
    let t = Instant::now();
    engine.wait_for_compaction().unwrap();
    println!("  Compaction settled in {:?}", t.elapsed());
    report_w("after concurrent", &engine);

    // Phase 7: Pure read phase — sustained reads, no writes
    phase_pure_reads(&engine, &ids);
    let t = Instant::now();
    engine.wait_for_compaction().unwrap();
    println!("  Compaction settled in {:?}", t.elapsed());
    report_w("after pure reads", &engine);

    // Summary
    println!("\n=== Stats ===");
    println!("SSTables on disk: {}", engine.sstable_count());
    println!("Secondary indexes: {}", engine.secondary_index_count());
    println!(
        "Effective W: {}, target W: {} (started at {})",
        engine.effective_w(),
        engine.target_w(),
        w
    );

    let cs = engine.compaction_stats();
    let runs = cs.runs.load(std::sync::atomic::Ordering::Relaxed);
    let bytes_r = cs.bytes_read.load(std::sync::atomic::Ordering::Relaxed);
    let bytes_w = cs.bytes_written.load(std::sync::atomic::Ordering::Relaxed);
    let ssts_r = cs.sstables_read.load(std::sync::atomic::Ordering::Relaxed);
    let ssts_w = cs
        .sstables_written
        .load(std::sync::atomic::Ordering::Relaxed);
    println!("\nCompaction:");
    println!("  Runs: {}", runs);
    println!("  SSTables read: {}, written: {}", ssts_r, ssts_w);
    println!(
        "  Bytes read: {:.1} MB, written: {:.1} MB",
        bytes_r as f64 / 1024.0 / 1024.0,
        bytes_w as f64 / 1024.0 / 1024.0
    );
    println!("  Write amplification: {:.2}x", cs.write_amplification());

    let stats = engine.query_stats();
    let patterns = stats.top_by_count(5);
    println!("\nTop query patterns:");
    for (pattern, ps) in &patterns {
        println!(
            "  {:>8} {:30} count={:<6} avg={:>8.2?} selectivity={:.3}",
            pattern.query_type,
            if pattern.filter_fields.is_empty() {
                "(no filter)".to_string()
            } else {
                pattern.filter_fields.join(",")
            },
            ps.count,
            ps.avg_latency(),
            ps.selectivity(),
        );
    }

    let candidates = stats.low_selectivity(0.1, 5);
    if !candidates.is_empty() {
        println!("\nIndex candidates (low selectivity):");
        for (pattern, ps) in &candidates {
            println!(
                "  fields={:?} selectivity={:.4} count={}",
                pattern.filter_fields,
                ps.selectivity(),
                ps.count
            );
        }
    }
}

/// Create a deterministic DocumentId from a counter.
/// Not a real UUIDv7, but deterministic and reproducible for benchmarks.
fn deterministic_id(i: u64) -> DocumentId {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&i.to_be_bytes());
    // Fill remaining bytes with a hash of i for uniqueness
    let hash = i.wrapping_mul(0x517cc1b727220a95);
    bytes[8..16].copy_from_slice(&hash.to_be_bytes());
    DocumentId::from_bytes(bytes)
}

fn make_product(i: u64) -> IBlob {
    make_product_with_id(deterministic_id(i), i)
}

fn make_product_with_id(id: DocumentId, i: u64) -> IBlob {
    let category = CATEGORIES[(i % CATEGORIES.len() as u64) as usize];
    let price = (i % 1000) as f64 + 0.99;
    let rating = (i % 50) as f64 / 10.0;
    IBlob::with_id(
        id,
        [
            ("type".into(), Value::String("product".into())),
            ("name".into(), Value::String(format!("Product #{i}"))),
            ("category".into(), Value::String(category.into())),
            ("price".into(), Value::F64(price)),
            ("rating".into(), Value::F64(rating)),
            ("stock".into(), Value::U64(i % 500)),
            (
                "description".into(),
                Value::String(format!("Description for product {i} in {category}")),
            ),
        ]
        .into(),
    )
}

fn phase_bulk_ingest(engine: &Arc<LsmEngine>) -> Vec<DocumentId> {
    const BATCH_SIZE: u64 = 1000;
    println!(
        "--- Phase 1: Bulk Ingest ({} products, batch={}) ---",
        NUM_PRODUCTS, BATCH_SIZE
    );

    let mut ids = Vec::with_capacity(NUM_PRODUCTS as usize);
    let start = Instant::now();

    let mut i = 0u64;
    while i < NUM_PRODUCTS {
        let end = (i + BATCH_SIZE).min(NUM_PRODUCTS);
        let batch: Vec<IBlob> = (i..end)
            .map(|j| {
                let blob = make_product(j);
                ids.push(*blob.id());
                blob
            })
            .collect();
        engine.put_batch(batch).unwrap();
        i = end;

        if i.is_multiple_of(10_000) {
            let elapsed = start.elapsed();
            let rate = i as f64 / elapsed.as_secs_f64();
            eprint!("\r  {}/{} ({:.0} docs/sec)", i, NUM_PRODUCTS, rate);
        }
    }

    let elapsed = start.elapsed();
    let rate = NUM_PRODUCTS as f64 / elapsed.as_secs_f64();
    println!(
        "\r  {} docs in {:.2?} ({:.0} docs/sec)",
        NUM_PRODUCTS, elapsed, rate
    );
    println!("  SSTables after ingest: {}", engine.sstable_count());

    ids
}

fn phase_random_updates(engine: &Arc<LsmEngine>, ids: &[DocumentId]) {
    let num_updates = NUM_PRODUCTS;
    println!(
        "\n--- Phase 1b: Random Updates ({} updates) ---",
        num_updates
    );

    // Single-threaded baseline
    {
        let start = Instant::now();
        for i in 0..num_updates {
            let idx = ((i
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407))
                % ids.len() as u64) as usize;
            let updated = make_product_with_id(ids[idx], i + NUM_PRODUCTS);
            engine.put(updated).unwrap();
        }
        let elapsed = start.elapsed();
        let rate = num_updates as f64 / elapsed.as_secs_f64();
        println!(
            "  1 thread:  {} updates in {:.2?} → {:.0} updates/sec",
            num_updates, elapsed, rate
        );
    }

    // Multi-threaded scaling
    for num_threads in [2, 4, 8] {
        let updates_per_thread = num_updates / num_threads as u64;
        let start = Instant::now();
        let mut handles = Vec::new();

        for t in 0..num_threads {
            let engine = Arc::clone(engine);
            let ids = ids.to_vec();
            handles.push(std::thread::spawn(move || {
                let offset = t as u64 * updates_per_thread;
                for i in 0..updates_per_thread {
                    let j = offset + i;
                    let idx = ((j
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407))
                        % ids.len() as u64) as usize;
                    let updated = make_product_with_id(ids[idx], j + NUM_PRODUCTS * 2);
                    engine.put(updated).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let elapsed = start.elapsed();
        let rate = num_updates as f64 / elapsed.as_secs_f64();
        println!(
            "  {} threads: {} updates in {:.2?} → {:.0} updates/sec",
            num_threads, num_updates, elapsed, rate
        );
    }

    println!("  SSTables after updates: {}", engine.sstable_count());
}

fn phase_point_lookups(engine: &Arc<LsmEngine>, ids: &[DocumentId]) {
    println!(
        "\n--- Phase 2: Point Lookups ({} random gets) ---",
        NUM_LOOKUPS
    );

    let mut latencies = Vec::with_capacity(NUM_LOOKUPS as usize);
    let mut found = 0u64;

    let start = Instant::now();
    for i in 0..NUM_LOOKUPS {
        // Pseudo-random selection using a simple hash
        let idx = ((i * 7919 + 1013) % ids.len() as u64) as usize;
        let t = Instant::now();
        if engine.get(&ids[idx]).unwrap().is_some() {
            found += 1;
        }
        latencies.push(t.elapsed());
    }
    let total = start.elapsed();

    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[latencies.len() * 95 / 100];
    let p99 = latencies[latencies.len() * 99 / 100];
    let rate = NUM_LOOKUPS as f64 / total.as_secs_f64();

    println!("  {}/{} found, {:.0} ops/sec", found, NUM_LOOKUPS, rate);
    println!(
        "  Latency: p50={:>8.2?}  p95={:>8.2?}  p99={:>8.2?}",
        p50, p95, p99
    );
}

fn phase_scan_queries(engine: &Arc<LsmEngine>) {
    println!("\n--- Phase 3: Scan + Sort Queries ---");

    // Query 1: filter by category, sort by price
    let filter = Filter::Eq {
        field: "category".into(),
        value: Value::String("electronics".into()),
    };
    let sort = vec![SortField {
        field: "price".into(),
        direction: SortDirection::Ascending,
    }];

    // First run — cold (no index)
    let t = Instant::now();
    let results = engine
        .execute(&Query::Scan {
            filter: Some(filter.clone()),
            sort: Some(sort.clone()),
            project: None,
            limit: None,
        })
        .unwrap();
    let cold = t.elapsed();
    println!("  category='electronics' ORDER BY price:");
    println!(
        "    Cold (no index): {} results in {:?}",
        results.len(),
        cold
    );

    // Second run — may use index if spilled
    let t = Instant::now();
    let results2 = engine
        .execute(&Query::Scan {
            filter: Some(filter.clone()),
            sort: Some(sort.clone()),
            project: None,
            limit: None,
        })
        .unwrap();
    let warm = t.elapsed();
    println!(
        "    Warm (with index): {} results in {:?}",
        results2.len(),
        warm
    );
    if cold > Duration::from_micros(1) {
        let speedup = cold.as_secs_f64() / warm.as_secs_f64().max(1e-9);
        println!("    Speedup: {:.1}x", speedup);
    }

    // Query 2: compound filter, sort descending
    let compound = Filter::And(vec![
        Filter::Gt {
            field: "price".into(),
            value: Value::F64(50.0),
        },
        Filter::Gt {
            field: "rating".into(),
            value: Value::F64(3.0),
        },
    ]);
    let sort_rating = vec![SortField {
        field: "rating".into(),
        direction: SortDirection::Descending,
    }];

    let t = Instant::now();
    let results = engine
        .execute(&Query::Scan {
            filter: Some(compound),
            sort: Some(sort_rating),
            project: None,
            limit: Some(20),
        })
        .unwrap();
    println!("\n  price>50 AND rating>3.0 ORDER BY rating DESC LIMIT 20:");
    println!("    {} results in {:?}", results.len(), t.elapsed());

    // Query 3: projection — only return name and price
    let t = Instant::now();
    let results = engine
        .execute(&Query::Scan {
            filter: Some(filter),
            sort: Some(sort),
            project: Some(vec!["name".into(), "price".into()]),
            limit: Some(10),
        })
        .unwrap();
    println!("\n  category='electronics' ORDER BY price LIMIT 10 PROJECT(name, price):");
    println!("    {} results in {:?}", results.len(), t.elapsed());
    for r in results.iter().take(3) {
        println!("    {:?} price={:?}", r.get("name"), r.get("price"));
    }

    println!(
        "  Secondary indexes after queries: {}",
        engine.secondary_index_count()
    );
}

fn phase_snapshot_reads(engine: &Arc<LsmEngine>, ids: &[DocumentId]) {
    println!("\n--- Phase 4: Snapshot Reads ---");

    // Take a snapshot of current state
    let snap = engine.snapshot();

    // Write some updates
    let num_updates = 1000u64;
    let t = Instant::now();
    for i in 0..num_updates {
        let idx = ((i * 3571) % ids.len() as u64) as usize;
        let updated = IBlob::with_id(
            ids[idx],
            [
                ("type".into(), Value::String("product".into())),
                ("name".into(), Value::String("UPDATED Product".to_string())),
                ("price".into(), Value::F64(999.99)),
                ("category".into(), Value::String("updated".into())),
                ("rating".into(), Value::F64(0.0)),
                ("stock".into(), Value::U64(0)),
                ("description".into(), Value::String("Updated".into())),
            ]
            .into(),
        );
        engine.put(updated).unwrap();
    }
    let write_time = t.elapsed();
    println!("  {} updates in {:?}", num_updates, write_time);

    // Read through snapshot — should see original values
    let t = Instant::now();
    let mut snapshot_correct = 0u64;
    for i in 0..100 {
        let idx = ((i * 3571) % ids.len() as u64) as usize;
        if let Some(blob) = snap.get(&ids[idx]).unwrap()
            && blob.get("name") != Some(&Value::String("UPDATED Product".into()))
        {
            snapshot_correct += 1;
        }
    }
    let snap_time = t.elapsed();
    println!("  100 snapshot reads in {:?}", snap_time);
    println!(
        "  Snapshot isolation: {}/100 reads saw pre-update data",
        snapshot_correct
    );

    // Read latest — should see updates
    let t = Instant::now();
    let mut latest_updated = 0u64;
    for i in 0..100 {
        let idx = ((i * 3571) % ids.len() as u64) as usize;
        if let Some(blob) = engine.get(&ids[idx]).unwrap()
            && blob.get("name") == Some(&Value::String("UPDATED Product".into()))
        {
            latest_updated += 1;
        }
    }
    let latest_time = t.elapsed();
    println!("  100 latest reads in {:?}", latest_time);
    println!("  Latest sees updates: {}/100", latest_updated);

    drop(snap);
    println!("  Snapshot released");
}

fn phase_mixed_workload(engine: &Arc<LsmEngine>, ids: &[DocumentId]) {
    println!("\n--- Phase 5: Mixed Read/Write ---");

    let total_ops = 10_000u64;
    let _write_ratio = 0.2; // 20% writes, 80% reads

    let start = Instant::now();
    let mut reads = 0u64;
    let mut writes = 0u64;

    for i in 0..total_ops {
        if (i % 5) == 0 {
            // Write (20%)
            engine.put(make_product(NUM_PRODUCTS + 10000 + i)).unwrap();
            writes += 1;
        } else {
            // Read — point lookup of real documents
            let idx = ((i * 7919) % ids.len() as u64) as usize;
            let _ = engine.get(&ids[idx]);
            reads += 1;
        }
    }

    let elapsed = start.elapsed();
    let rate = total_ops as f64 / elapsed.as_secs_f64();
    println!(
        "  {} ops ({} reads, {} writes) in {:.2?}",
        total_ops, reads, writes, elapsed
    );
    println!("  {:.0} ops/sec", rate);
}

fn phase_concurrent_reads(engine: &Arc<LsmEngine>, ids: &[DocumentId]) {
    println!("\n--- Phase 6: Concurrent Reads ---");

    let ops_per_thread = 10_000u64;

    for num_threads in [1, 2, 4, 8] {
        let start = Instant::now();
        let mut handles = Vec::new();

        for t in 0..num_threads {
            let engine = Arc::clone(engine);
            let ids: Vec<DocumentId> = ids.to_vec();
            handles.push(std::thread::spawn(move || {
                let mut found = 0u64;
                for i in 0..ops_per_thread {
                    let idx = ((i * 7919 + t as u64 * 1013) % ids.len() as u64) as usize;
                    if engine.get(&ids[idx]).unwrap().is_some() {
                        found += 1;
                    }
                }
                found
            }));
        }

        let total_found: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let elapsed = start.elapsed();
        let total_ops = ops_per_thread * num_threads as u64;
        let rate = total_ops as f64 / elapsed.as_secs_f64();

        println!(
            "  {:>2} threads: {:>6} ops in {:>8.2?} → {:>8.0} ops/sec  ({} found)",
            num_threads, total_ops, elapsed, rate, total_found
        );
    }

    // Also test concurrent reads + writes
    println!("\n  Concurrent reads + writes (4 reader threads, 1 writer thread):");
    let ops_per_reader = 10_000u64;
    let ops_writer = 2_000u64;

    let start = Instant::now();
    let mut handles = Vec::new();

    // 4 reader threads
    for t in 0..4u32 {
        let engine = Arc::clone(engine);
        let ids: Vec<DocumentId> = ids.to_vec();
        handles.push(std::thread::spawn(move || {
            let mut found = 0u64;
            for i in 0..ops_per_reader {
                let idx = ((i * 7919 + t as u64 * 1013) % ids.len() as u64) as usize;
                if engine.get(&ids[idx]).unwrap().is_some() {
                    found += 1;
                }
            }
            (found, 0u64) // (reads, writes)
        }));
    }

    // 1 writer thread
    {
        let engine = Arc::clone(engine);
        handles.push(std::thread::spawn(move || {
            for i in 0..ops_writer {
                let blob = make_product(NUM_PRODUCTS + 100_000 + i);
                engine.put(blob).unwrap();
            }
            (0u64, ops_writer) // (reads, writes)
        }));
    }

    let mut total_reads = 0u64;
    let mut total_writes = 0u64;
    for h in handles {
        let (r, w) = h.join().unwrap();
        total_reads += r;
        total_writes += w;
    }
    let elapsed = start.elapsed();
    let total_ops = total_reads + total_writes;
    let rate = total_ops as f64 / elapsed.as_secs_f64();
    println!(
        "    {} total ops ({} reads, {} writes) in {:?} → {:.0} ops/sec",
        total_ops, total_reads, total_writes, elapsed, rate
    );
}

fn phase_pure_reads(engine: &Arc<LsmEngine>, ids: &[DocumentId]) {
    let num_reads = NUM_PRODUCTS * 2;
    println!(
        "\n--- Phase 7: Pure Reads ({} sequential gets, zero writes) ---",
        num_reads
    );

    let start = Instant::now();
    let mut found = 0u64;

    for i in 0..num_reads {
        let idx = ((i * 7919 + 2027) % ids.len() as u64) as usize;
        if engine.get(&ids[idx]).unwrap().is_some() {
            found += 1;
        }
    }

    let elapsed = start.elapsed();
    let rate = num_reads as f64 / elapsed.as_secs_f64();
    println!(
        "  {}/{} found, {:.0} ops/sec in {:?}",
        found, num_reads, rate, elapsed
    );
}
