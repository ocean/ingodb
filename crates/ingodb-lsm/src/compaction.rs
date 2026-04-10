use ingodb_blob::{DocumentId, IBlob};
use std::path::PathBuf;

/// Action to take on a blob during compaction.
pub enum CompactionAction {
    /// Keep the blob as-is
    Keep,
    /// Drop the blob (e.g., tombstone purge)
    Drop,
    /// Replace with a transformed blob (e.g., semantic shredding)
    Transform(IBlob),
}

/// Pluggable filter applied during compaction.
/// This is the extension point for future adaptive morphing.
pub trait CompactionFilter {
    fn filter(&mut self, id: &DocumentId, blob: &IBlob) -> CompactionAction;
}

/// Metadata about an SSTable needed for compaction decisions.
#[derive(Debug, Clone)]
pub struct SstMeta {
    pub path: PathBuf,
    pub min_id: DocumentId,
    pub max_id: DocumentId,
    pub file_size: u64,
    /// Index in the sstables list (for tracking creation order within a level)
    pub seq: u64,
}

// ---------------------------------------------------------------------------
// UCS-inspired compaction
// ---------------------------------------------------------------------------

/// Unified Compaction Strategy (UCS-inspired).
///
/// Assigns SSTables to levels based on file size, detects overlapping
/// SSTables within each level, and triggers compaction when the overlap
/// count reaches the threshold.
///
/// Key formulas (from Cassandra UCS):
/// - `fanout f = if W < 0 { 2 - W } else { 2 + W }`
/// - `threshold t = if W <= 0 { 2 } else { f }`
/// - `level(sst) = floor(log_f(file_size / flush_size)).max(0)`
///
/// Scaling parameter W controls the read/write amplification tradeoff:
/// - W < 0: leveled (low read amp, high write amp)
/// - W = 0: balanced (f=2, t=2)
/// - W > 0: tiered (low write amp, high read amp)
pub struct UcsCompaction {
    /// Scaling parameter W
    scaling_parameter: i32,
    /// Expected memtable flush size in bytes
    flush_size: u64,
}

impl UcsCompaction {
    pub fn new(scaling_parameter: i32, flush_size: u64) -> Self {
        UcsCompaction {
            scaling_parameter,
            flush_size: flush_size.max(1),
        }
    }

    /// Fanout factor f.
    pub fn fanout(&self) -> f64 {
        if self.scaling_parameter < 0 {
            (2 - self.scaling_parameter) as f64
        } else {
            (2 + self.scaling_parameter) as f64
        }
    }

    /// Compaction trigger threshold t.
    pub fn threshold(&self) -> usize {
        if self.scaling_parameter <= 0 {
            2
        } else {
            self.fanout() as usize
        }
    }

    /// Compute the level for an SSTable based on its file size.
    pub fn level_for_size(&self, file_size: u64) -> u32 {
        if file_size <= self.flush_size {
            return 0;
        }
        let f = self.fanout();
        (file_size as f64 / self.flush_size as f64)
            .log(f)
            .floor()
            .max(0.0) as u32
    }

    /// Pick SSTables to compact based on UCS overlap detection.
    ///
    /// Returns the paths to compact and the output level, or None.
    pub fn pick_compaction(&self, sstables: &[SstMeta]) -> Option<CompactionPick> {
        if sstables.is_empty() {
            return None;
        }

        let t = self.threshold();

        // Assign levels
        let leveled: Vec<(u32, &SstMeta)> = sstables
            .iter()
            .map(|s| (self.level_for_size(s.file_size), s))
            .collect();

        // Find max level across all SSTables
        let max_level = leveled.iter().map(|(l, _)| *l).max().unwrap_or(0);

        // Group by level
        let mut levels: std::collections::BTreeMap<u32, Vec<&SstMeta>> =
            std::collections::BTreeMap::new();
        for (level, meta) in &leveled {
            levels.entry(*level).or_default().push(*meta);
        }

        // Check each level (lowest first — reduces read amp fastest)
        for (&level, ssts) in &levels {
            if let Some(group) = find_overlap_group(ssts, t) {
                let output_level = level + 1;
                return Some(CompactionPick {
                    inputs: group.iter().map(|s| s.path.clone()).collect(),
                    output_level,
                    max_level,
                });
            }
        }

        None
    }

    /// Pick ALL eligible compaction groups across all levels.
    /// Returns multiple independent compaction jobs that can run in parallel.
    pub fn pick_all_compactions(&self, sstables: &[SstMeta]) -> Vec<CompactionPick> {
        if sstables.is_empty() {
            return Vec::new();
        }

        let t = self.threshold();

        let leveled: Vec<(u32, &SstMeta)> = sstables
            .iter()
            .map(|s| (self.level_for_size(s.file_size), s))
            .collect();

        let max_level = leveled.iter().map(|(l, _)| *l).max().unwrap_or(0);

        let mut levels: std::collections::BTreeMap<u32, Vec<&SstMeta>> =
            std::collections::BTreeMap::new();
        for (level, meta) in &leveled {
            levels.entry(*level).or_default().push(*meta);
        }

        let mut picks = Vec::new();
        for (&level, ssts) in &levels {
            // Find ALL non-overlapping groups at this level that meet threshold
            let mut remaining: Vec<&SstMeta> = ssts.clone();
            while let Some(group) = find_overlap_group(&remaining, t) {
                let group_paths: std::collections::HashSet<PathBuf> =
                    group.iter().map(|s| s.path.clone()).collect();
                picks.push(CompactionPick {
                    inputs: group.iter().map(|s| s.path.clone()).collect(),
                    output_level: level + 1,
                    max_level,
                });
                // Remove the picked group from remaining
                remaining.retain(|s| !group_paths.contains(&s.path));
            }
        }
        picks
    }
}

/// Result of a compaction pick.
#[derive(Debug)]
pub struct CompactionPick {
    /// SSTable paths to compact
    pub inputs: Vec<PathBuf>,
    /// Level the output SSTable will belong to
    pub output_level: u32,
    /// Highest level among all current SSTables
    pub max_level: u32,
}

/// Find a group of overlapping SSTables that reaches the threshold.
///
/// Sort by min_id, sweep with running max_id to find overlap chains.
fn find_overlap_group<'a>(ssts: &[&'a SstMeta], threshold: usize) -> Option<Vec<&'a SstMeta>> {
    if ssts.len() < threshold {
        return None;
    }

    let mut sorted: Vec<&SstMeta> = ssts.to_vec();
    sorted.sort_by_key(|s| s.min_id);

    let mut best_group: Option<Vec<&SstMeta>> = None;
    let mut current_group: Vec<&SstMeta> = vec![sorted[0]];
    let mut running_max = sorted[0].max_id;

    for &sst in &sorted[1..] {
        if sst.min_id <= running_max {
            // Overlaps with current group
            current_group.push(sst);
            if sst.max_id > running_max {
                running_max = sst.max_id;
            }
        } else {
            // No overlap — check if current group qualifies
            if current_group.len() >= threshold
                && best_group
                    .as_ref()
                    .is_none_or(|g| current_group.len() > g.len())
            {
                best_group = Some(current_group.clone());
            }
            current_group = vec![sst];
            running_max = sst.max_id;
        }
    }

    // Check final group
    if current_group.len() >= threshold
        && best_group
            .as_ref()
            .is_none_or(|g| current_group.len() > g.len())
    {
        best_group = Some(current_group);
    }

    best_group
}

// ---------------------------------------------------------------------------
// Tombstone filter
// ---------------------------------------------------------------------------

/// Purges tombstones during compaction when safe to do so.
///
/// A tombstone can be safely purged when the compaction output is at the
/// last (highest) level — there's no older SSTable below that could hold
/// a shadowed live entry. Full compaction is a special case of this.
pub struct TombstoneFilter {
    /// Whether tombstones should be purged in this compaction
    pub purge: bool,
}

impl TombstoneFilter {
    /// Create a filter that purges tombstones if the output level is the max level
    /// AND no snapshots are active (conservative: don't purge while any snapshot exists).
    pub fn new(output_level: u32, max_level: u32, has_active_snapshots: bool) -> Self {
        TombstoneFilter {
            purge: output_level > max_level && !has_active_snapshots,
        }
    }
}

impl CompactionFilter for TombstoneFilter {
    fn filter(&mut self, _id: &DocumentId, blob: &IBlob) -> CompactionAction {
        if self.purge && blob.is_deleted() {
            CompactionAction::Drop
        } else {
            CompactionAction::Keep
        }
    }
}

// ---------------------------------------------------------------------------
// Size-Tiered Compaction (legacy fallback)
// ---------------------------------------------------------------------------

/// Size-Tiered Compaction Strategy.
///
/// Groups SSTables by similar file size and triggers compaction
/// when a group reaches the threshold.
pub struct SizeTieredCompaction {
    threshold: usize,
}

impl SizeTieredCompaction {
    pub fn new(threshold: usize) -> Self {
        SizeTieredCompaction { threshold }
    }

    /// Pick SSTables to compact, if any group has reached the threshold.
    /// Returns the paths of SSTables to merge, or None.
    pub fn pick_compaction(&self, sst_paths: &[PathBuf]) -> Option<Vec<PathBuf>> {
        if sst_paths.len() < self.threshold {
            return None;
        }

        // Get file sizes
        let mut sized: Vec<(PathBuf, u64)> = sst_paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok().map(|m| (p.clone(), m.len())))
            .collect();

        sized.sort_by_key(|(_, size)| *size);

        // Group by similar size (within 2x of each other)
        let mut groups: Vec<Vec<PathBuf>> = Vec::new();
        let mut current_group: Vec<(PathBuf, u64)> = Vec::new();

        for (path, size) in sized {
            if current_group.is_empty() {
                current_group.push((path, size));
            } else {
                let group_min = current_group[0].1;
                if size <= group_min * 2 {
                    current_group.push((path, size));
                } else {
                    if current_group.len() >= self.threshold {
                        groups.push(current_group.iter().map(|(p, _)| p.clone()).collect());
                    }
                    current_group = vec![(path, size)];
                }
            }
        }
        if current_group.len() >= self.threshold {
            groups.push(current_group.iter().map(|(p, _)| p.clone()).collect());
        }

        // Return the first group that reached threshold
        groups.into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_compaction_needed() {
        let compaction = SizeTieredCompaction::new(4);
        let paths = vec![PathBuf::from("a.sst"), PathBuf::from("b.sst")];
        assert!(compaction.pick_compaction(&paths).is_none());
    }

    #[test]
    fn test_ucs_level_computation() {
        // W=0: f=2, flush_size=1000
        let ucs = UcsCompaction::new(0, 1000);
        assert_eq!(ucs.fanout(), 2.0);
        assert_eq!(ucs.threshold(), 2);

        assert_eq!(ucs.level_for_size(500), 0); // below flush size
        assert_eq!(ucs.level_for_size(1000), 0); // exactly flush size
        assert_eq!(ucs.level_for_size(1999), 0); // < 2 * flush
        assert_eq!(ucs.level_for_size(2000), 1); // 2x = L1
        assert_eq!(ucs.level_for_size(4000), 2); // 4x = L2
        assert_eq!(ucs.level_for_size(8000), 3); // 8x = L3
    }

    #[test]
    fn test_ucs_tiered_mode() {
        // W=2: f=4, t=4
        let ucs = UcsCompaction::new(2, 1000);
        assert_eq!(ucs.fanout(), 4.0);
        assert_eq!(ucs.threshold(), 4);

        assert_eq!(ucs.level_for_size(1000), 0);
        assert_eq!(ucs.level_for_size(4000), 1);
        assert_eq!(ucs.level_for_size(16000), 2);
    }

    #[test]
    fn test_ucs_leveled_mode() {
        // W=-2: f=4, t=2
        let ucs = UcsCompaction::new(-2, 1000);
        assert_eq!(ucs.fanout(), 4.0);
        assert_eq!(ucs.threshold(), 2);
    }

    fn make_meta(path: &str, min: u8, max: u8, size: u64, seq: u64) -> SstMeta {
        SstMeta {
            path: PathBuf::from(path),
            min_id: DocumentId::from_bytes([min; 16]),
            max_id: DocumentId::from_bytes([max; 16]),
            file_size: size,
            seq,
        }
    }

    #[test]
    fn test_ucs_overlap_detection() {
        // W=0: f=2, t=2, flush_size=1000
        let ucs = UcsCompaction::new(0, 1000);

        // Two L0 SSTables that overlap
        let metas = vec![
            make_meta("a.sst", 0x10, 0x50, 800, 1),
            make_meta("b.sst", 0x30, 0x80, 900, 2),
        ];

        let pick = ucs.pick_compaction(&metas);
        assert!(pick.is_some());
        let pick = pick.unwrap();
        assert_eq!(pick.inputs.len(), 2);
        assert_eq!(pick.output_level, 1);
    }

    #[test]
    fn test_ucs_no_overlap_no_compaction() {
        let ucs = UcsCompaction::new(0, 1000);

        // Two L0 SSTables that don't overlap
        let metas = vec![
            make_meta("a.sst", 0x10, 0x20, 800, 1),
            make_meta("b.sst", 0x30, 0x40, 900, 2),
        ];

        assert!(ucs.pick_compaction(&metas).is_none());
    }

    #[test]
    fn test_ucs_prefers_lowest_level() {
        // W=0: f=2, t=2
        let ucs = UcsCompaction::new(0, 1000);

        // Two overlapping L0 and two overlapping L1
        let metas = vec![
            make_meta("a.sst", 0x10, 0x50, 800, 1),  // L0
            make_meta("b.sst", 0x30, 0x80, 900, 2),  // L0
            make_meta("c.sst", 0x10, 0x50, 2500, 3), // L1
            make_meta("d.sst", 0x30, 0x80, 3000, 4), // L1
        ];

        let pick = ucs.pick_compaction(&metas).unwrap();
        // Should pick L0 (lowest level)
        assert!(pick.inputs.contains(&PathBuf::from("a.sst")));
        assert!(pick.inputs.contains(&PathBuf::from("b.sst")));
        assert_eq!(pick.output_level, 1);
    }

    #[test]
    fn test_tombstone_filter_purges_at_max_level() {
        let id = DocumentId::new();
        let tomb = IBlob::tombstone(id);
        let live = IBlob::from_pairs(vec![("x", ingodb_blob::Value::U64(1))]);

        // output_level > max_level → purge
        let mut filter = TombstoneFilter::new(3, 2, false);
        assert!(matches!(filter.filter(&id, &tomb), CompactionAction::Drop));
        assert!(matches!(filter.filter(&id, &live), CompactionAction::Keep));

        // output_level <= max_level → keep tombstone
        let mut filter = TombstoneFilter::new(1, 2, false);
        assert!(matches!(filter.filter(&id, &tomb), CompactionAction::Keep));
    }
}
