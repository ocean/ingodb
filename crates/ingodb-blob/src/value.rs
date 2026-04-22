use crate::DocumentId;

/// Tagged value types supported by IngoDB.
///
/// These are the "standard library" of primitives that the engine understands
/// natively for indexing and comparison. Everything else is opaque bytes.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Null / missing value
    Null,
    /// Boolean
    Bool(bool),
    /// Signed 64-bit integer
    I64(i64),
    /// Unsigned 64-bit integer
    U64(u64),
    /// 64-bit floating point
    F64(f64),
    /// UTF-8 string
    String(String),
    /// Raw bytes (opaque to the engine)
    Bytes(Vec<u8>),
    /// UUID value (16 bytes). Often used to store document IDs as field values.
    Uuid(DocumentId),
    /// Ordered list of values
    Array(Vec<Value>),
    /// Nested document (sorted key-value pairs)
    Document(Vec<(String, Value)>),
}

// Type tags for binary encoding
const TAG_NULL: u8 = 0;
const TAG_BOOL: u8 = 1;
const TAG_I64: u8 = 2;
const TAG_U64: u8 = 3;
const TAG_F64: u8 = 4;
const TAG_STRING: u8 = 5;
const TAG_BYTES: u8 = 6;
const TAG_UUID: u8 = 7;
const TAG_ARRAY: u8 = 8;
const TAG_DOCUMENT: u8 = 9;

impl Value {
    /// Encode this value into the buffer, returning bytes written.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            Value::Null => buf.push(TAG_NULL),
            Value::Bool(b) => {
                buf.push(TAG_BOOL);
                buf.push(if *b { 1 } else { 0 });
            }
            Value::I64(v) => {
                buf.push(TAG_I64);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Value::U64(v) => {
                buf.push(TAG_U64);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Value::F64(v) => {
                buf.push(TAG_F64);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Value::String(s) => {
                buf.push(TAG_STRING);
                buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
            Value::Bytes(b) => {
                buf.push(TAG_BYTES);
                buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
                buf.extend_from_slice(b);
            }
            Value::Uuid(id) => {
                buf.push(TAG_UUID);
                buf.extend_from_slice(id.as_bytes());
            }
            Value::Array(arr) => {
                buf.push(TAG_ARRAY);
                buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                for v in arr {
                    v.encode(buf);
                }
            }
            Value::Document(fields) => {
                buf.push(TAG_DOCUMENT);
                buf.extend_from_slice(&(fields.len() as u32).to_le_bytes());
                for (key, val) in fields {
                    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
                    buf.extend_from_slice(key.as_bytes());
                    val.encode(buf);
                }
            }
        }
    }

    /// Decode a value from the buffer at the given offset. Returns (value, bytes_consumed).
    pub fn decode(buf: &[u8], offset: usize) -> Result<(Value, usize), crate::BlobError> {
        if offset >= buf.len() {
            return Err(crate::BlobError::BufferTooShort {
                need: offset + 1,
                have: buf.len(),
            });
        }

        let tag = buf[offset];
        let pos = offset + 1;

        match tag {
            TAG_NULL => Ok((Value::Null, 1)),
            TAG_BOOL => {
                check_len(buf, pos, 1)?;
                let v = buf[pos] != 0;
                Ok((Value::Bool(v), 2))
            }
            TAG_I64 => {
                check_len(buf, pos, 8)?;
                let v = i64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
                Ok((Value::I64(v), 9))
            }
            TAG_U64 => {
                check_len(buf, pos, 8)?;
                let v = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
                Ok((Value::U64(v), 9))
            }
            TAG_F64 => {
                check_len(buf, pos, 8)?;
                let v = f64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
                Ok((Value::F64(v), 9))
            }
            TAG_STRING => {
                check_len(buf, pos, 4)?;
                let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
                let pos = pos + 4;
                check_len(buf, pos, len)?;
                let s = std::str::from_utf8(&buf[pos..pos + len])?.to_string();
                Ok((Value::String(s), 5 + len))
            }
            TAG_BYTES => {
                check_len(buf, pos, 4)?;
                let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
                let pos = pos + 4;
                check_len(buf, pos, len)?;
                let b = buf[pos..pos + len].to_vec();
                Ok((Value::Bytes(b), 5 + len))
            }
            TAG_UUID => {
                check_len(buf, pos, 16)?;
                let mut id_bytes = [0u8; 16];
                id_bytes.copy_from_slice(&buf[pos..pos + 16]);
                Ok((Value::Uuid(DocumentId::from_bytes(id_bytes)), 17))
            }
            TAG_ARRAY => {
                check_len(buf, pos, 4)?;
                let count = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
                let mut pos = pos + 4;
                let mut arr = Vec::with_capacity(count);
                let mut total = 5; // tag + count
                for _ in 0..count {
                    let (v, consumed) = Value::decode(buf, pos)?;
                    pos += consumed;
                    total += consumed;
                    arr.push(v);
                }
                Ok((Value::Array(arr), total))
            }
            TAG_DOCUMENT => {
                check_len(buf, pos, 4)?;
                let count = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
                let mut pos = pos + 4;
                let mut fields = Vec::with_capacity(count);
                let mut total = 5; // tag + count
                for _ in 0..count {
                    check_len(buf, pos, 4)?;
                    let klen = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
                    pos += 4;
                    total += 4;
                    check_len(buf, pos, klen)?;
                    let key = std::str::from_utf8(&buf[pos..pos + klen])?.to_string();
                    pos += klen;
                    total += klen;
                    let (val, consumed) = Value::decode(buf, pos)?;
                    pos += consumed;
                    total += consumed;
                    fields.push((key, val));
                }
                Ok((Value::Document(fields), total))
            }
            _ => Err(crate::BlobError::UnknownValueType(tag)),
        }
    }
}

fn check_len(buf: &[u8], offset: usize, need: usize) -> Result<(), crate::BlobError> {
    if offset + need > buf.len() {
        Err(crate::BlobError::BufferTooShort {
            need: offset + need,
            have: buf.len(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: &Value) {
        let mut buf = Vec::new();
        v.encode(&mut buf);
        let (decoded, consumed) = Value::decode(&buf, 0).unwrap();
        assert_eq!(&decoded, v);
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn test_null() {
        roundtrip(&Value::Null);
    }

    #[test]
    fn test_bool() {
        roundtrip(&Value::Bool(true));
        roundtrip(&Value::Bool(false));
    }

    #[test]
    fn test_integers() {
        roundtrip(&Value::I64(-42));
        roundtrip(&Value::I64(i64::MIN));
        roundtrip(&Value::U64(42));
        roundtrip(&Value::U64(u64::MAX));
    }

    #[test]
    fn test_f64() {
        roundtrip(&Value::F64(1.23456));
        roundtrip(&Value::F64(f64::NEG_INFINITY));
    }

    #[test]
    fn test_string() {
        roundtrip(&Value::String("hello IngoDB".into()));
        roundtrip(&Value::String(String::new()));
        roundtrip(&Value::String("mörkkö 🦀".into()));
    }

    #[test]
    fn test_bytes() {
        roundtrip(&Value::Bytes(vec![0, 1, 2, 255]));
        roundtrip(&Value::Bytes(vec![]));
    }

    #[test]
    fn test_ref() {
        roundtrip(&Value::Uuid(DocumentId::from_bytes([0xAB; 16])));
    }

    #[test]
    fn test_array() {
        roundtrip(&Value::Array(vec![
            Value::I64(1),
            Value::String("two".into()),
            Value::Null,
        ]));
    }

    #[test]
    fn test_nested_document() {
        roundtrip(&Value::Document(vec![
            ("name".into(), Value::String("Henrik".into())),
            (
                "scores".into(),
                Value::Array(vec![Value::U64(100), Value::U64(200)]),
            ),
            (
                "address".into(),
                Value::Document(vec![
                    ("city".into(), Value::String("Helsinki".into())),
                    ("zip".into(), Value::String("00100".into())),
                ]),
            ),
        ]));
    }
}
