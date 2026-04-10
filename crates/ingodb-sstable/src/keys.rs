use ingodb_blob::{IBlob, Value};

/// Extracts a comparable sort key from an IBlob.
///
/// The extracted key bytes must produce the desired sort order when compared
/// lexicographically. The default implementation (`IdKeyExtractor`) sorts
/// by `_id`. Custom implementations can sort by field values or arbitrary
/// user-defined criteria.
pub trait KeyExtractor: Send + Sync {
    /// Extract the sort key as bytes.
    fn extract_key(&self, blob: &IBlob) -> Vec<u8>;
}

/// Default key extractor: sorts by `_id` (DocumentId bytes).
pub struct IdKeyExtractor;

impl KeyExtractor for IdKeyExtractor {
    fn extract_key(&self, blob: &IBlob) -> Vec<u8> {
        blob.id().as_bytes().to_vec()
    }
}

/// MVCC key extractor: sorts by `(_id, _version)` — 32 bytes total.
/// This enables multiple versions of the same document in one SSTable,
/// with natural binary search on the composite key.
pub struct MvccKeyExtractor;

impl KeyExtractor for MvccKeyExtractor {
    fn extract_key(&self, blob: &IBlob) -> Vec<u8> {
        let mut key = Vec::with_capacity(32);
        key.extend_from_slice(blob.id().as_bytes());
        key.extend_from_slice(blob.version().as_bytes());
        key
    }
}

/// Key extractor that sorts by specified field values.
/// Multi-field keys are the concatenation of each field's comparable encoding.
pub struct FieldKeyExtractor {
    fields: Vec<String>,
}

impl FieldKeyExtractor {
    pub fn new(fields: Vec<String>) -> Self {
        FieldKeyExtractor { fields }
    }
}

impl KeyExtractor for FieldKeyExtractor {
    fn extract_key(&self, blob: &IBlob) -> Vec<u8> {
        let mut key = Vec::new();
        for field in &self.fields {
            match blob.get_field(field) {
                Some(val) => key.extend_from_slice(&encode_comparable_value(&val)),
                None => key.push(0xFF), // null sorts last
            }
        }
        key
    }
}

/// Encode a Value into bytes that preserve ordering under lexicographic comparison.
///
/// Type tags ensure values of different types don't accidentally compare equal.
/// Within each type, encoding is chosen so that byte comparison matches value comparison.
pub fn encode_comparable_value(value: &Value) -> Vec<u8> {
    let mut buf = Vec::new();
    match value {
        Value::Null => buf.push(0x00),
        Value::Bool(b) => {
            buf.push(0x01);
            buf.push(if *b { 1 } else { 0 });
        }
        Value::I64(n) => {
            buf.push(0x02);
            // Flip sign bit so negative < positive in unsigned byte comparison
            buf.extend_from_slice(&((*n as u64) ^ 0x8000_0000_0000_0000).to_be_bytes());
        }
        Value::U64(n) => {
            buf.push(0x03);
            buf.extend_from_slice(&n.to_be_bytes());
        }
        Value::F64(f) => {
            buf.push(0x04);
            // IEEE 754 float-to-sortable-bytes encoding
            let bits = f.to_bits();
            let sortable = if bits & 0x8000_0000_0000_0000 != 0 {
                !bits // negative: flip all bits
            } else {
                bits ^ 0x8000_0000_0000_0000 // positive: flip sign bit
            };
            buf.extend_from_slice(&sortable.to_be_bytes());
        }
        Value::String(s) => {
            buf.push(0x05);
            buf.extend_from_slice(s.as_bytes());
            buf.push(0x00); // null terminator for proper prefix ordering
        }
        Value::Bytes(b) => {
            buf.push(0x06);
            buf.extend_from_slice(b);
            buf.push(0x00);
        }
        Value::Uuid(id) => {
            buf.push(0x07);
            buf.extend_from_slice(id.as_bytes());
        }
        Value::Array(_) | Value::Document(_) => {
            // Complex types: not meaningfully sortable, use hash for determinism
            buf.push(0x08);
            let mut hasher_buf = Vec::new();
            value.encode(&mut hasher_buf);
            let h = blake3::hash(&hasher_buf);
            buf.extend_from_slice(&h.as_bytes()[..16]);
        }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingodb_blob::DocumentId;

    #[test]
    fn test_i64_ordering() {
        let vals = [i64::MIN, -100, -1, 0, 1, 100, i64::MAX];
        let encoded: Vec<Vec<u8>> = vals
            .iter()
            .map(|n| encode_comparable_value(&Value::I64(*n)))
            .collect();
        for i in 1..encoded.len() {
            assert!(
                encoded[i - 1] < encoded[i],
                "i64 order: {} should sort before {}",
                vals[i - 1],
                vals[i]
            );
        }
    }

    #[test]
    fn test_f64_ordering() {
        let vals = [f64::NEG_INFINITY, -1.0, 0.0, 1.0, f64::INFINITY];
        let encoded: Vec<Vec<u8>> = vals
            .iter()
            .map(|n| encode_comparable_value(&Value::F64(*n)))
            .collect();
        for i in 1..encoded.len() {
            assert!(
                encoded[i - 1] < encoded[i],
                "f64 order: {} should sort before {}",
                vals[i - 1],
                vals[i]
            );
        }
    }

    #[test]
    fn test_different_types_dont_collide() {
        let a = encode_comparable_value(&Value::U64(0));
        let b = encode_comparable_value(&Value::I64(0));
        let c = encode_comparable_value(&Value::String("0".into()));
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn test_field_key_multi_field() {
        let extractor = FieldKeyExtractor::new(vec!["a".into(), "b".into()]);
        let blob1 = IBlob::from_pairs(vec![("a", Value::String("x".into())), ("b", Value::U64(1))]);
        let blob2 = IBlob::from_pairs(vec![("a", Value::String("x".into())), ("b", Value::U64(2))]);
        let blob3 = IBlob::from_pairs(vec![("a", Value::String("y".into())), ("b", Value::U64(0))]);
        let k1 = extractor.extract_key(&blob1);
        let k2 = extractor.extract_key(&blob2);
        let k3 = extractor.extract_key(&blob3);
        assert!(k1 < k2, "same first field, second field determines order");
        assert!(k2 < k3, "different first field dominates");
    }

    #[test]
    fn test_id_key_extractor() {
        let blob = IBlob::from_pairs(vec![("x", Value::Null)]);
        let key = IdKeyExtractor.extract_key(&blob);
        assert_eq!(key, blob.id().as_bytes().to_vec());
    }
}
