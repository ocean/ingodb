mod bloom;
mod error;
mod keys;
mod reader;
mod writer;

pub use bloom::BloomFilter;
pub use error::SSTableError;
pub use keys::{
    FieldKeyExtractor, IdKeyExtractor, KeyExtractor, MvccKeyExtractor, encode_comparable_value,
};
pub use reader::SSTableReader;
pub use writer::SSTableWriter;

/// SSTable file magic: "ISST"
pub const SSTABLE_MAGIC: [u8; 4] = *b"ISST";

/// SSTable format version
pub const SSTABLE_VERSION: u16 = 2;

/// Default data block size (4 KB, aligned for direct I/O)
pub const DEFAULT_BLOCK_SIZE: usize = 4096;

/// Footer size: index_offset(8) + index_count(4) + bloom_offset(8) + bloom_size(4) + magic(4) + version(2) = 30
pub const FOOTER_SIZE: usize = 30;

#[cfg(test)]
mod tests {
    use super::*;
    use ingodb_blob::{IBlob, Value};

    fn make_entries(n: usize) -> Vec<IBlob> {
        (0..n)
            .map(|i| {
                IBlob::from_pairs(vec![
                    ("id", Value::U64(i as u64)),
                    ("name", Value::String(format!("entry-{i}"))),
                ])
            })
            .collect()
    }

    #[test]
    fn test_write_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.sst");

        let mut blobs = make_entries(100);
        let extractor = IdKeyExtractor;

        SSTableWriter::new()
            .write(&path, &mut blobs, &extractor)
            .unwrap();
        let reader = SSTableReader::open(&path).unwrap();

        // Verify all entries can be looked up by _id
        for blob in &blobs {
            let key = extractor.extract_key(blob);
            let found = reader.get(&key).unwrap().expect("entry not found");
            assert_eq!(found.id(), blob.id());
            assert_eq!(found.fields(), blob.fields());
        }

        // Verify a missing key returns None
        let missing_key = IdKeyExtractor.extract_key(&IBlob::from_pairs(vec![("x", Value::Null)]));
        assert!(reader.get(&missing_key).unwrap().is_none());
    }

    #[test]
    fn test_iter_all_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("iter.sst");

        let mut blobs = make_entries(50);
        SSTableWriter::new()
            .write(&path, &mut blobs, &IdKeyExtractor)
            .unwrap();
        let reader = SSTableReader::open(&path).unwrap();

        let all = reader.iter().unwrap();
        assert_eq!(all.len(), 50);

        // Verify sorted order by key
        for i in 1..all.len() {
            assert!(all[i - 1].0 <= all[i].0);
        }
    }

    #[test]
    fn test_small_block_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small_blocks.sst");

        let mut blobs = make_entries(20);

        SSTableWriter::with_block_size(128)
            .write(&path, &mut blobs, &IdKeyExtractor)
            .unwrap();
        let reader = SSTableReader::open(&path).unwrap();

        assert!(reader.block_count() > 1, "expected multiple blocks");

        for blob in &blobs {
            let key = IdKeyExtractor.extract_key(blob);
            let found = reader.get(&key).unwrap().expect("entry not found");
            assert_eq!(found.id(), blob.id());
        }
    }

    #[test]
    fn test_single_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single.sst");

        let mut blobs = make_entries(1);
        SSTableWriter::new()
            .write(&path, &mut blobs, &IdKeyExtractor)
            .unwrap();
        let reader = SSTableReader::open(&path).unwrap();

        let key = IdKeyExtractor.extract_key(&blobs[0]);
        let found = reader.get(&key).unwrap().unwrap();
        assert_eq!(found.id(), blobs[0].id());
    }

    #[test]
    fn test_min_max_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("minmax.sst");

        let mut blobs = make_entries(50);
        SSTableWriter::new()
            .write(&path, &mut blobs, &IdKeyExtractor)
            .unwrap();
        let reader = SSTableReader::open(&path).unwrap();

        let all = reader.iter().unwrap();
        assert_eq!(reader.min_key(), all.first().unwrap().0.as_slice());
        assert_eq!(reader.max_key(), all.last().unwrap().0.as_slice());
    }

    #[test]
    fn test_min_max_single_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("minmax1.sst");

        let mut blobs = make_entries(1);
        SSTableWriter::new()
            .write(&path, &mut blobs, &IdKeyExtractor)
            .unwrap();
        let reader = SSTableReader::open(&path).unwrap();

        assert_eq!(reader.min_key(), reader.max_key());
    }

    #[test]
    fn test_field_key_extractor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("field_sort.sst");

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

        let extractor = FieldKeyExtractor::new(vec!["name".into()]);
        SSTableWriter::new()
            .write(&path, &mut blobs, &extractor)
            .unwrap();
        let reader = SSTableReader::open(&path).unwrap();

        // Iterate — should be sorted by name
        let all = reader.iter().unwrap();
        let names: Vec<_> = all
            .iter()
            .map(|(_, blob)| blob.get("name").cloned())
            .collect();
        assert_eq!(
            names,
            vec![
                Some(Value::String("Alice".into())),
                Some(Value::String("Bob".into())),
                Some(Value::String("Charlie".into())),
            ]
        );

        // Point lookup by key
        let alice_key = encode_comparable_value(&Value::String("Alice".into()));
        let found = reader.get(&alice_key).unwrap().unwrap();
        assert_eq!(found.get("name"), Some(&Value::String("Alice".into())));
    }

    #[test]
    fn test_comparable_encoding_order() {
        // Verify that byte encoding preserves value ordering
        let vals = [
            Value::U64(0),
            Value::U64(1),
            Value::U64(255),
            Value::U64(256),
            Value::U64(u64::MAX),
        ];
        let encoded: Vec<Vec<u8>> = vals.iter().map(encode_comparable_value).collect();
        for i in 1..encoded.len() {
            assert!(
                encoded[i - 1] < encoded[i],
                "encoding should preserve order: {:?} < {:?}",
                vals[i - 1],
                vals[i]
            );
        }

        let strings = [
            Value::String("".into()),
            Value::String("a".into()),
            Value::String("ab".into()),
            Value::String("b".into()),
        ];
        let encoded: Vec<Vec<u8>> = strings.iter().map(encode_comparable_value).collect();
        for i in 1..encoded.len() {
            assert!(
                encoded[i - 1] < encoded[i],
                "string encoding should preserve order: {:?} < {:?}",
                strings[i - 1],
                strings[i]
            );
        }
    }
}
