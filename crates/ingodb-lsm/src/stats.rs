use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Identifies a query pattern for aggregation.
/// Two queries with the same pattern shape get aggregated together.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryPattern {
    /// Query type: "get", "scan", "traverse"
    pub query_type: String,
    /// Fields used in filter predicates (sorted for deterministic hashing)
    pub filter_fields: Vec<String>,
    /// Fields used for sorting (sorted by appearance order)
    pub sort_fields: Vec<String>,
    /// Join edge for traverse queries: (from_field, to_field)
    pub join_edge: Option<(String, String)>,
}

/// Metrics recorded for a single query execution.
#[derive(Debug, Clone)]
pub struct QueryMetrics {
    /// The query pattern
    pub pattern: QueryPattern,
    /// Wall-clock latency of the query
    pub latency: Duration,
    /// Number of documents examined (before filtering)
    pub docs_scanned: u64,
    /// Number of documents returned (after filtering)
    pub docs_returned: u64,
}

/// Aggregated statistics for a query pattern.
#[derive(Debug, Clone)]
pub struct PatternStats {
    /// Number of times this pattern was executed
    pub count: u64,
    /// Total latency across all executions
    pub total_latency: Duration,
    /// Total documents scanned
    pub total_scanned: u64,
    /// Total documents returned
    pub total_returned: u64,
}

impl PatternStats {
    fn new() -> Self {
        PatternStats {
            count: 0,
            total_latency: Duration::ZERO,
            total_scanned: 0,
            total_returned: 0,
        }
    }

    fn record(&mut self, metrics: &QueryMetrics) {
        self.count += 1;
        self.total_latency += metrics.latency;
        self.total_scanned += metrics.docs_scanned;
        self.total_returned += metrics.docs_returned;
    }

    /// Average latency per query.
    pub fn avg_latency(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total_latency / self.count as u32
        }
    }

    /// Selectivity: fraction of scanned documents that were returned.
    /// Low selectivity (close to 0) means a secondary index would help.
    pub fn selectivity(&self) -> f64 {
        if self.total_scanned == 0 {
            1.0
        } else {
            self.total_returned as f64 / self.total_scanned as f64
        }
    }
}

/// Thread-safe query statistics collector.
///
/// Records per-query metrics and aggregates them by query pattern.
/// Designed to be always-on with minimal overhead.
pub struct QueryStats {
    patterns: Mutex<HashMap<QueryPattern, PatternStats>>,
}

impl QueryStats {
    pub fn new() -> Self {
        QueryStats {
            patterns: Mutex::new(HashMap::new()),
        }
    }

    /// Record metrics for a completed query.
    pub fn record(&self, metrics: QueryMetrics) {
        let mut patterns = self.patterns.lock();
        patterns
            .entry(metrics.pattern.clone())
            .or_insert_with(PatternStats::new)
            .record(&metrics);
    }

    /// Get aggregated stats for all observed patterns.
    pub fn all_patterns(&self) -> Vec<(QueryPattern, PatternStats)> {
        self.patterns
            .lock()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Get stats for a specific pattern, if it exists.
    pub fn get_pattern(&self, pattern: &QueryPattern) -> Option<PatternStats> {
        self.patterns.lock().get(pattern).cloned()
    }

    /// Get the top N patterns by execution count.
    pub fn top_by_count(&self, n: usize) -> Vec<(QueryPattern, PatternStats)> {
        let mut all = self.all_patterns();
        all.sort_by(|a, b| b.1.count.cmp(&a.1.count));
        all.truncate(n);
        all
    }

    /// Get patterns with low selectivity (potential index candidates).
    /// Returns patterns where selectivity < threshold and count >= min_count.
    pub fn low_selectivity(
        &self,
        threshold: f64,
        min_count: u64,
    ) -> Vec<(QueryPattern, PatternStats)> {
        self.all_patterns()
            .into_iter()
            .filter(|(_, stats)| stats.count >= min_count && stats.selectivity() < threshold)
            .collect()
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        self.patterns.lock().clear();
    }
}

/// Helper to time a query and build metrics.
pub struct QueryTimer {
    pattern: QueryPattern,
    start: Instant,
    docs_scanned: u64,
}

impl QueryTimer {
    pub fn start(pattern: QueryPattern) -> Self {
        QueryTimer {
            pattern,
            start: Instant::now(),
            docs_scanned: 0,
        }
    }

    /// Record the number of documents scanned (before filtering).
    pub fn set_docs_scanned(&mut self, count: u64) {
        self.docs_scanned = count;
    }

    /// Finish timing and produce metrics.
    pub fn finish(self, docs_returned: u64) -> QueryMetrics {
        QueryMetrics {
            pattern: self.pattern,
            latency: self.start.elapsed(),
            docs_scanned: self.docs_scanned,
            docs_returned,
        }
    }
}

/// Extract filter field names from a Filter tree.
pub fn extract_filter_fields(filter: &ingodb_query::Filter) -> Vec<String> {
    let mut fields = Vec::new();
    collect_filter_fields(filter, &mut fields);
    fields.sort();
    fields.dedup();
    fields
}

fn collect_filter_fields(filter: &ingodb_query::Filter, fields: &mut Vec<String>) {
    use ingodb_query::Filter;
    match filter {
        Filter::Eq { field, .. }
        | Filter::Gt { field, .. }
        | Filter::Lt { field, .. }
        | Filter::Range { field, .. }
        | Filter::Exists { field } => {
            fields.push(field.clone());
        }
        Filter::And(filters) | Filter::Or(filters) => {
            for f in filters {
                collect_filter_fields(f, fields);
            }
        }
        Filter::Not(f) => {
            collect_filter_fields(f, fields);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_pattern(filter_fields: &[&str], sort_fields: &[&str]) -> QueryPattern {
        QueryPattern {
            query_type: "scan".into(),
            filter_fields: filter_fields.iter().map(|s| s.to_string()).collect(),
            sort_fields: sort_fields.iter().map(|s| s.to_string()).collect(),
            join_edge: None,
        }
    }

    #[test]
    fn test_record_and_retrieve() {
        let stats = QueryStats::new();
        let pattern = scan_pattern(&["age"], &[]);

        stats.record(QueryMetrics {
            pattern: pattern.clone(),
            latency: Duration::from_millis(10),
            docs_scanned: 1000,
            docs_returned: 50,
        });

        let ps = stats.get_pattern(&pattern).unwrap();
        assert_eq!(ps.count, 1);
        assert_eq!(ps.total_scanned, 1000);
        assert_eq!(ps.total_returned, 50);
    }

    #[test]
    fn test_aggregation() {
        let stats = QueryStats::new();
        let pattern = scan_pattern(&["name"], &[]);

        for _ in 0..10 {
            stats.record(QueryMetrics {
                pattern: pattern.clone(),
                latency: Duration::from_millis(5),
                docs_scanned: 100,
                docs_returned: 10,
            });
        }

        let ps = stats.get_pattern(&pattern).unwrap();
        assert_eq!(ps.count, 10);
        assert_eq!(ps.total_latency, Duration::from_millis(50));
        assert_eq!(ps.total_scanned, 1000);
        assert_eq!(ps.total_returned, 100);
        assert_eq!(ps.avg_latency(), Duration::from_millis(5));
    }

    #[test]
    fn test_selectivity() {
        let stats = QueryStats::new();
        let pattern = scan_pattern(&["email"], &[]);

        stats.record(QueryMetrics {
            pattern: pattern.clone(),
            latency: Duration::from_millis(100),
            docs_scanned: 10_000,
            docs_returned: 1,
        });

        let ps = stats.get_pattern(&pattern).unwrap();
        assert!(
            ps.selectivity() < 0.001,
            "very low selectivity — index candidate"
        );
    }

    #[test]
    fn test_low_selectivity_detection() {
        let stats = QueryStats::new();

        // Pattern 1: high selectivity (returns most docs)
        let p1 = scan_pattern(&["type"], &[]);
        for _ in 0..5 {
            stats.record(QueryMetrics {
                pattern: p1.clone(),
                latency: Duration::from_millis(10),
                docs_scanned: 100,
                docs_returned: 80,
            });
        }

        // Pattern 2: low selectivity (returns few docs — index candidate)
        let p2 = scan_pattern(&["email"], &[]);
        for _ in 0..5 {
            stats.record(QueryMetrics {
                pattern: p2.clone(),
                latency: Duration::from_millis(50),
                docs_scanned: 10_000,
                docs_returned: 1,
            });
        }

        let candidates = stats.low_selectivity(0.01, 3);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, p2);
    }

    #[test]
    fn test_top_by_count() {
        let stats = QueryStats::new();

        let p1 = scan_pattern(&["a"], &[]);
        let p2 = scan_pattern(&["b"], &[]);

        for _ in 0..10 {
            stats.record(QueryMetrics {
                pattern: p1.clone(),
                latency: Duration::from_millis(1),
                docs_scanned: 10,
                docs_returned: 5,
            });
        }
        for _ in 0..3 {
            stats.record(QueryMetrics {
                pattern: p2.clone(),
                latency: Duration::from_millis(1),
                docs_scanned: 10,
                docs_returned: 5,
            });
        }

        let top = stats.top_by_count(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, p1);
        assert_eq!(top[0].1.count, 10);
    }

    #[test]
    fn test_distinct_patterns() {
        let stats = QueryStats::new();

        // Same query type but different filter fields = different patterns
        let p1 = scan_pattern(&["age"], &[]);
        let p2 = scan_pattern(&["name"], &[]);

        stats.record(QueryMetrics {
            pattern: p1.clone(),
            latency: Duration::from_millis(1),
            docs_scanned: 10,
            docs_returned: 5,
        });
        stats.record(QueryMetrics {
            pattern: p2.clone(),
            latency: Duration::from_millis(1),
            docs_scanned: 10,
            docs_returned: 5,
        });

        assert_eq!(stats.all_patterns().len(), 2);
    }

    #[test]
    fn test_traverse_pattern() {
        let stats = QueryStats::new();
        let pattern = QueryPattern {
            query_type: "traverse".into(),
            filter_fields: vec!["type".into()],
            sort_fields: vec![],
            join_edge: Some(("user_id".into(), "_id".into())),
        };

        stats.record(QueryMetrics {
            pattern: pattern.clone(),
            latency: Duration::from_millis(20),
            docs_scanned: 500,
            docs_returned: 3,
        });

        let ps = stats.get_pattern(&pattern).unwrap();
        assert_eq!(ps.count, 1);
        assert!(ps.selectivity() < 0.01);
    }

    #[test]
    fn test_extract_filter_fields() {
        use ingodb_blob::Value;
        use ingodb_query::Filter;

        let filter = Filter::And(vec![
            Filter::Eq {
                field: "type".into(),
                value: Value::String("user".into()),
            },
            Filter::Gt {
                field: "age".into(),
                value: Value::U64(30),
            },
        ]);

        let fields = extract_filter_fields(&filter);
        assert_eq!(fields, vec!["age", "type"]);
    }

    #[test]
    fn test_query_timer() {
        let pattern = scan_pattern(&["x"], &["y"]);
        let mut timer = QueryTimer::start(pattern);
        timer.set_docs_scanned(100);
        std::thread::sleep(Duration::from_millis(1));
        let metrics = timer.finish(10);
        assert!(metrics.latency >= Duration::from_millis(1));
        assert_eq!(metrics.docs_scanned, 100);
        assert_eq!(metrics.docs_returned, 10);
        assert_eq!(metrics.pattern.sort_fields, vec!["y"]);
    }

    #[test]
    fn test_reset() {
        let stats = QueryStats::new();
        stats.record(QueryMetrics {
            pattern: scan_pattern(&["a"], &[]),
            latency: Duration::from_millis(1),
            docs_scanned: 10,
            docs_returned: 5,
        });
        assert_eq!(stats.all_patterns().len(), 1);
        stats.reset();
        assert_eq!(stats.all_patterns().len(), 0);
    }
}
