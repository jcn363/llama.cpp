//! GGUF (GGML Universal File) format reader and writer.
//!
//! This crate provides safe Rust bindings for reading and writing GGUF files,
//! the binary format used by llama.cpp for storing model weights and metadata.
//!
//! # GGUF File Format (v3)
//!
//! ```text
//! 1. Magic "GGUF" (4 bytes)
//! 2. Version (u32)
//! 3. Tensor count (i64)
//! 4. KV pair count (i64)
//! 5. KV pairs: key (string), type (i32), value
//! 6. Tensor info: name (string), n_dims (u32), dims (i64 × n), type (i32), offset (u64)
//! 7. Tensor data blob (aligned to general.alignment, default 32)
//! ```
//!
//! # Example
//!
//! ```no_run
//! use gguf::GgufReader;
//!
//! let reader = GgufReader::from_file("model.gguf").unwrap();
//! println!("Tensors: {}", reader.tensor_count());
//! println!("KV pairs: {}", reader.metadata_count());
//! for i in 0..reader.tensor_count() as usize {
//!     let info = reader.tensor_info(i).unwrap();
//!     println!("  {} {:?}", info.name, info.shape);
//! }
//! ```

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]

use std::io;
use std::path::Path;

use thiserror::Error;

// ─── Constants ───────────────────────────────────────────────────────────────

/// GGUF magic bytes: "GGUF" in little-endian u32.
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Supported GGUF version.
pub const GGUF_VERSION: u32 = 3;

/// Default alignment for tensor data.
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// ─── GGUF Value Types ────────────────────────────────────────────────────────

/// GGUF metadata value types (13 total).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum GgufType {
    /// Unsigned 8-bit integer.
    Uint8 = 0,
    /// Signed 8-bit integer.
    Int8 = 1,
    /// Unsigned 16-bit integer.
    Uint16 = 2,
    /// Signed 16-bit integer.
    Int16 = 3,
    /// Unsigned 32-bit integer.
    Uint32 = 4,
    /// Signed 32-bit integer.
    Int32 = 5,
    /// 32-bit IEEE 754 float.
    Float32 = 6,
    /// Boolean (stored as int8).
    Bool = 7,
    /// UTF-8 string.
    String = 8,
    /// Array of homogeneous values.
    Array = 9,
    /// Unsigned 64-bit integer.
    Uint64 = 10,
    /// Signed 64-bit integer.
    Int64 = 11,
    /// 64-bit IEEE 754 float.
    Float64 = 12,
}

impl GgufType {
    /// Returns the size in bytes of a single value of this type.
    /// Returns 0 for variable-size types (String, Array).
    #[must_use]
    pub fn size_of(self) -> usize {
        match self {
            GgufType::Uint8 | GgufType::Int8 | GgufType::Bool => 1,
            GgufType::Uint16 | GgufType::Int16 => 2,
            GgufType::Uint32 | GgufType::Int32 | GgufType::Float32 => 4,
            GgufType::Uint64 | GgufType::Int64 | GgufType::Float64 => 8,
            GgufType::String | GgufType::Array => 0,
        }
    }

    /// Returns the human-readable name of this type.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GgufType::Uint8 => "u8",
            GgufType::Int8 => "i8",
            GgufType::Uint16 => "u16",
            GgufType::Int16 => "i16",
            GgufType::Uint32 => "u32",
            GgufType::Int32 => "i32",
            GgufType::Float32 => "f32",
            GgufType::Bool => "bool",
            GgufType::String => "str",
            GgufType::Array => "arr",
            GgufType::Uint64 => "u64",
            GgufType::Int64 => "i64",
            GgufType::Float64 => "f64",
        }
    }

    /// Try to convert from a raw i32 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not a valid GGUF type.
    pub fn from_i32(v: i32) -> Result<Self, GgufError> {
        match v {
            0 => Ok(GgufType::Uint8),
            1 => Ok(GgufType::Int8),
            2 => Ok(GgufType::Uint16),
            3 => Ok(GgufType::Int16),
            4 => Ok(GgufType::Uint32),
            5 => Ok(GgufType::Int32),
            6 => Ok(GgufType::Float32),
            7 => Ok(GgufType::Bool),
            8 => Ok(GgufType::String),
            9 => Ok(GgufType::Array),
            10 => Ok(GgufType::Uint64),
            11 => Ok(GgufType::Int64),
            12 => Ok(GgufType::Float64),
            _ => Err(GgufError::DecodeError(format!("unknown gguf_type: {v}"))),
        }
    }
}

// ─── GGML Tensor Types ───────────────────────────────────────────────────────

/// GGML tensor data types (stored as i32 in GGUF files).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum GgmlType {
    /// 32-bit float.
    F32 = 0,
    /// 16-bit float.
    F16 = 1,
    /// 4-bit quantized (variant 0).
    Q4_0 = 2,
    /// 4-bit quantized (variant 1).
    Q4_1 = 3,
    /// 5-bit quantized (variant 0).
    Q5_0 = 6,
    /// 5-bit quantized (variant 1).
    Q5_1 = 7,
    /// 8-bit quantized (variant 0).
    Q8_0 = 8,
    /// 8-bit quantized (variant 1).
    Q8_1 = 9,
    /// 2-bit K-quants.
    Q2_K = 10,
    /// 3-bit K-quants.
    Q3_K = 11,
    /// 4-bit K-quants.
    Q4_K = 12,
    /// 5-bit K-quants.
    Q5_K = 13,
    /// 6-bit K-quants.
    Q6_K = 14,
    /// 8-bit K-quants.
    Q8_K = 15,
    /// IQ2 XXS.
    Iq2Xxs = 16,
    /// IQ2 XS.
    Iq2Xs = 17,
    /// IQ3 XXS.
    Iq3Xxs = 18,
    /// IQ1 S.
    Iq1S = 19,
    /// IQ4 NL.
    Iq4Nl = 20,
    /// IQ3 S.
    Iq3S = 21,
    /// IQ2 S.
    Iq2S = 22,
    /// IQ4 XS.
    Iq4Xs = 23,
    /// 8-bit integer.
    I8 = 24,
    /// 16-bit integer.
    I16 = 25,
    /// 32-bit integer.
    I32 = 26,
    /// 64-bit integer.
    I64 = 27,
    /// 64-bit float.
    F64 = 28,
    /// IQ1 M.
    Iq1M = 29,
    /// Brain float 16.
    Bf16 = 30,
    /// Ternary quantized 1.0.
    Tq1_0 = 34,
    /// Ternary quantized 2.0.
    Tq2_0 = 35,
    /// MXFP4.
    Mxfp4 = 39,
    /// NVFP4.
    Nvfp4 = 40,
    /// 1-bit quantized.
    Q1_0 = 41,
}

impl GgmlType {
    /// Try to convert from a raw i32 value.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not a valid GGML type.
    pub fn from_i32(v: i32) -> Result<Self, GgufError> {
        match v {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            9 => Ok(GgmlType::Q8_1),
            10 => Ok(GgmlType::Q2_K),
            11 => Ok(GgmlType::Q3_K),
            12 => Ok(GgmlType::Q4_K),
            13 => Ok(GgmlType::Q5_K),
            14 => Ok(GgmlType::Q6_K),
            15 => Ok(GgmlType::Q8_K),
            16 => Ok(GgmlType::Iq2Xxs),
            17 => Ok(GgmlType::Iq2Xs),
            18 => Ok(GgmlType::Iq3Xxs),
            19 => Ok(GgmlType::Iq1S),
            20 => Ok(GgmlType::Iq4Nl),
            21 => Ok(GgmlType::Iq3S),
            22 => Ok(GgmlType::Iq2S),
            23 => Ok(GgmlType::Iq4Xs),
            24 => Ok(GgmlType::I8),
            25 => Ok(GgmlType::I16),
            26 => Ok(GgmlType::I32),
            27 => Ok(GgmlType::I64),
            28 => Ok(GgmlType::F64),
            29 => Ok(GgmlType::Iq1M),
            30 => Ok(GgmlType::Bf16),
            34 => Ok(GgmlType::Tq1_0),
            35 => Ok(GgmlType::Tq2_0),
            39 => Ok(GgmlType::Mxfp4),
            40 => Ok(GgmlType::Nvfp4),
            41 => Ok(GgmlType::Q1_0),
            _ => Err(GgufError::DecodeError(format!("unknown ggml_type: {v}"))),
        }
    }

    /// Returns the human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "f32",
            GgmlType::F16 => "f16",
            GgmlType::Q4_0 => "q4_0",
            GgmlType::Q4_1 => "q4_1",
            GgmlType::Q5_0 => "q5_0",
            GgmlType::Q5_1 => "q5_1",
            GgmlType::Q8_0 => "q8_0",
            GgmlType::Q8_1 => "q8_1",
            GgmlType::Q2_K => "q2_k",
            GgmlType::Q3_K => "q3_k",
            GgmlType::Q4_K => "q4_k",
            GgmlType::Q5_K => "q5_k",
            GgmlType::Q6_K => "q6_k",
            GgmlType::Q8_K => "q8_k",
            GgmlType::Iq2Xxs => "iq2_xxs",
            GgmlType::Iq2Xs => "iq2_xs",
            GgmlType::Iq3Xxs => "iq3_xxs",
            GgmlType::Iq1S => "iq1_s",
            GgmlType::Iq4Nl => "iq4_nl",
            GgmlType::Iq3S => "iq3_s",
            GgmlType::Iq2S => "iq2_s",
            GgmlType::Iq4Xs => "iq4_xs",
            GgmlType::I8 => "i8",
            GgmlType::I16 => "i16",
            GgmlType::I32 => "i32",
            GgmlType::I64 => "i64",
            GgmlType::F64 => "f64",
            GgmlType::Iq1M => "iq1_m",
            GgmlType::Bf16 => "bf16",
            GgmlType::Tq1_0 => "tq1_0",
            GgmlType::Tq2_0 => "tq2_0",
            GgmlType::Mxfp4 => "mxfp4",
            GgmlType::Nvfp4 => "nvfp4",
            GgmlType::Q1_0 => "q1_0",
        }
    }
}

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors that can occur when reading or writing GGUF files.
#[derive(Debug, Error)]
pub enum GgufError {
    /// The file could not be opened or read.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// The file does not have a valid GGUF magic number.
    #[error("invalid GGUF magic number")]
    InvalidMagic,

    /// The GGUF version is not supported.
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    /// A value could not be decoded.
    #[error("decode error: {0}")]
    DecodeError(String),

    /// The file is truncated or incomplete.
    #[error("unexpected end of file")]
    UnexpectedEof,

    /// The tensor index is out of range.
    #[error("tensor index {0} out of range (max {1})")]
    TensorIndexOutOfRange(usize, usize),

    /// The KV key index is out of range.
    #[error("KV key index {0} out of range (max {1})")]
    KvIndexOutOfRange(usize, usize),
}

/// Result type alias for GGUF operations.
pub type GgufResult<T> = Result<T, GgufError>;

// ─── GGUF Value ──────────────────────────────────────────────────────────────

/// A single GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// 32-bit float.
    F32(f32),
    /// Boolean.
    Bool(bool),
    /// String.
    Str(String),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 64-bit float.
    F64(f64),
    /// Array of values.
    Array {
        /// Element type.
        elem_type: GgufType,
        /// Elements.
        data: Vec<GgufValue>,
    },
}

impl GgufValue {
    /// Returns the GGUF type of this value.
    #[must_use]
    pub fn gguf_type(&self) -> GgufType {
        match self {
            GgufValue::U8(_) => GgufType::Uint8,
            GgufValue::I8(_) => GgufType::Int8,
            GgufValue::U16(_) => GgufType::Uint16,
            GgufValue::I16(_) => GgufType::Int16,
            GgufValue::U32(_) => GgufType::Uint32,
            GgufValue::I32(_) => GgufType::Int32,
            GgufValue::F32(_) => GgufType::Float32,
            GgufValue::Bool(_) => GgufType::Bool,
            GgufValue::Str(_) => GgufType::String,
            GgufValue::U64(_) => GgufType::Uint64,
            GgufValue::I64(_) => GgufType::Int64,
            GgufValue::F64(_) => GgufType::Float64,
            GgufValue::Array { .. } => GgufType::Array,
        }
    }
}

// ─── Tensor Info ─────────────────────────────────────────────────────────────

/// Information about a single tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// Shape (dimensions in reverse order from file, matching ggml convention).
    pub shape: Vec<i64>,
    /// Data type.
    pub dtype: GgmlType,
    /// Offset into the tensor data blob.
    pub offset: u64,
}

impl TensorInfo {
    /// De‑quantize the raw tensor bytes into a `Vec<f32>`.
    /// Supports F32, F16, and common quantization types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K-Q6_K).
    ///
    /// # Panics
    ///
    /// This function will panic if the underlying byte slices cannot be
    /// converted to the expected array sizes via `try_into`. The panic is
    /// considered acceptable because the size checks are performed just
    /// before the conversion, and a mismatch would indicate a corrupted
    /// GGUF file.
    ///
    /// # Errors
    ///
    /// Returns a `GgufError::DecodeError` if the tensor size is not a multiple
    /// of the element size, or if the dtype is unsupported.
    pub fn dequantize(&self, raw: &[u8]) -> Result<Vec<f32>, GgufError> {
        match self.dtype {
            GgmlType::F32 => {
                // raw length should be a multiple of 4
                if raw.len() % 4 != 0 {
                    return Err(GgufError::DecodeError("F32 tensor size not multiple of 4".into()));
                }
                let num_elements = raw.len() / 4;
                
                // Parallel for large tensors (> 64K elements)
                if num_elements > 65536 {
                    use rayon::prelude::*;
                    let mut out = vec![0.0f32; num_elements];
                    out.par_chunks_mut(1024)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let start = chunk_idx * 1024 * 4;
                            for (i, out_val) in chunk.iter_mut().enumerate() {
                                let byte_idx = start + i * 4;
                                *out_val = f32::from_le_bytes(raw[byte_idx..byte_idx + 4].try_into().unwrap());
                            }
                        });
                    Ok(out)
                } else {
                    let mut out = Vec::with_capacity(num_elements);
                    for chunk in raw.chunks_exact(4) {
                        let v = f32::from_le_bytes(chunk.try_into().unwrap());
                        out.push(v);
                    }
                    Ok(out)
                }
            }
            GgmlType::F16 => {
                // each f16 is 2 bytes
                if raw.len() % 2 != 0 {
                    return Err(GgufError::DecodeError("F16 tensor size not multiple of 2".into()));
                }
                let num_elements = raw.len() / 2;
                
                // Parallel for large tensors (> 64K elements)
                if num_elements > 65536 {
                    use rayon::prelude::*;
                    let mut out = vec![0.0f32; num_elements];
                    out.par_chunks_mut(1024)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let start = chunk_idx * 1024 * 2;
                            for (i, out_val) in chunk.iter_mut().enumerate() {
                                let byte_idx = start + i * 2;
                                let bits = u16::from_le_bytes(raw[byte_idx..byte_idx + 2].try_into().unwrap());
                                *out_val = half::f16::from_bits(bits).to_f32();
                            }
                        });
                    Ok(out)
                } else {
                    let mut out = Vec::with_capacity(num_elements);
                    for chunk in raw.chunks_exact(2) {
                        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                        let f: f32 = half::f16::from_bits(bits).to_f32();
                        out.push(f);
                    }
                    Ok(out)
                }
            }
            GgmlType::Q4_0 => dequantize_q4_0(raw),
            GgmlType::Q4_1 => dequantize_q4_1(raw),
            GgmlType::Q5_0 => dequantize_q5_0(raw),
            GgmlType::Q5_1 => dequantize_q5_1(raw),
            GgmlType::Q8_0 => dequantize_q8_0(raw),
            GgmlType::Q2_K => dequantize_q2_k(raw),
            GgmlType::Q3_K => dequantize_q3_k(raw),
            GgmlType::Q4_K => dequantize_q4_k(raw),
            GgmlType::Q5_K => dequantize_q5_k(raw),
            GgmlType::Q6_K => dequantize_q6_k(raw),
            _ => Err(GgufError::DecodeError(format!("unsupported dtype for dequantize: {:?}", self.dtype))),
        }
    }
}

// ─── Dequantization Functions ────────────────────────────────────────────────

/// Q4_0: 4-bit quantization, variant 0.
/// Block size: 32 elements.
/// Layout: [d: f16][qs: 16 bytes]
/// Each byte in qs contains 2 4-bit values: qs[i] & 0xF, qs[i] >> 4
fn dequantize_q4_0(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK4_0: usize = 32;
    const BLOCK_SIZE: usize = 2 + 16; // d (2 bytes) + qs (16 bytes)
    
    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError("Q4_0 tensor size not multiple of block size".into()));
    }
    
    let num_blocks = raw.len() / BLOCK_SIZE;
    
    // Parallel for large tensors (> 64K elements = 2048 blocks)
    if num_blocks > 2048 {
        use rayon::prelude::*;
        let mut out = vec![0.0f32; num_blocks * QK4_0];
        
        out.par_chunks_mut(1024)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_block = (chunk_idx * 1024) / QK4_0;
                let end_block = ((chunk_idx + 1) * 1024) / QK4_0;
                
                for block_idx in start_block..end_block.min(num_blocks) {
                    let block_start = block_idx * BLOCK_SIZE;
                    let block = &raw[block_start..block_start + BLOCK_SIZE];
                    let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
                    let qs = &block[2..18];
                    
                    let out_start = block_idx * QK4_0 - start_block * QK4_0;
                    for i in 0..16 {
                        let v0 = (qs[i] & 0x0F) as i8 - 8;
                        let v1 = (qs[i] >> 4) as i8 - 8;
                        if out_start + i * 2 < chunk.len() {
                            chunk[out_start + i * 2] = v0 as f32 * d;
                            if out_start + i * 2 + 1 < chunk.len() {
                                chunk[out_start + i * 2 + 1] = v1 as f32 * d;
                            }
                        }
                    }
                }
            });
        
        Ok(out)
    } else {
        let mut out = Vec::with_capacity(num_blocks * QK4_0);
        
        for block in raw.chunks_exact(BLOCK_SIZE) {
            let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
            let qs = &block[2..18];
            
            for i in 0..16 {
                let v0 = (qs[i] & 0x0F) as i8 - 8;
                let v1 = (qs[i] >> 4) as i8 - 8;
                out.push(v0 as f32 * d);
                out.push(v1 as f32 * d);
            }
        }
        
        Ok(out)
    }
}

/// Q4_1: 4-bit quantization, variant 1.
/// Block size: 32 elements.
/// Layout: [d: f16][m: f16][qs: 16 bytes]
/// value = d * qs + m
fn dequantize_q4_1(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK4_1: usize = 32;
    const BLOCK_SIZE: usize = 2 + 2 + 16; // d (2) + m (2) + qs (16)
    
    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError("Q4_1 tensor size not multiple of block size".into()));
    }
    
    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK4_1);
    
    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
        let m = half::f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap())).to_f32();
        let qs = &block[4..20];
        
        for i in 0..16 {
            let v0 = (qs[i] & 0x0F) as f32;
            let v1 = (qs[i] >> 4) as f32;
            out.push(v0 * d + m);
            out.push(v1 * d + m);
        }
    }
    
    Ok(out)
}

/// Q5_0: 5-bit quantization, variant 0.
/// Block size: 32 elements.
/// Layout: [d: f16][qh: 4 bytes][qs: 16 bytes]
fn dequantize_q5_0(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK5_0: usize = 32;
    const BLOCK_SIZE: usize = 2 + 4 + 16; // d (2) + qh (4) + qs (16)
    
    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError("Q5_0 tensor size not multiple of block size".into()));
    }
    
    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK5_0);
    
    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
        let qh = &block[2..6];
        let qs = &block[6..22];
        
        for i in 0..16 {
            let h0 = ((qh[i / 4] >> ((i % 4) * 2)) & 0x01) as i8;
            let h1 = ((qh[i / 4] >> ((i % 4) * 2 + 1)) & 0x01) as i8;
            
            let v0 = ((qs[i] & 0x0F) | ((h0 as u8) << 4)) as i8 - 16;
            let v1 = ((qs[i] >> 4) | ((h1 as u8) << 4)) as i8 - 16;
            out.push(v0 as f32 * d);
            out.push(v1 as f32 * d);
        }
    }
    
    Ok(out)
}

/// Q5_1: 5-bit quantization, variant 1.
/// Block size: 32 elements.
/// Layout: [d: f16][m: f16][qh: 4 bytes][qs: 16 bytes]
fn dequantize_q5_1(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK5_1: usize = 32;
    const BLOCK_SIZE: usize = 2 + 2 + 4 + 16; // d (2) + m (2) + qh (4) + qs (16)
    
    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError("Q5_1 tensor size not multiple of block size".into()));
    }
    
    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK5_1);
    
    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
        let m = half::f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap())).to_f32();
        let qh = &block[4..8];
        let qs = &block[8..24];
        
        for i in 0..16 {
            let h0 = ((qh[i / 4] >> ((i % 4) * 2)) & 0x01) as u8;
            let h1 = ((qh[i / 4] >> ((i % 4) * 2 + 1)) & 0x01) as u8;
            
            let v0 = ((qs[i] & 0x0F) | (h0 << 4)) as f32;
            let v1 = ((qs[i] >> 4) | (h1 << 4)) as f32;
            out.push(v0 * d + m);
            out.push(v1 * d + m);
        }
    }
    
    Ok(out)
}

/// Q8_0: 8-bit quantization, variant 0.
/// Block size: 32 elements.
/// Layout: [d: f16][qs: 32 bytes]
fn dequantize_q8_0(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK8_0: usize = 32;
    const BLOCK_SIZE: usize = 2 + 32; // d (2) + qs (32)
    
    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError("Q8_0 tensor size not multiple of block size".into()));
    }
    
    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK8_0);
    
    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
        let qs = &block[2..34];
        
        for &q in qs {
            out.push(q as i8 as f32 * d);
        }
    }
    
    Ok(out)
}

/// Q2_K: 2-bit K-quant.
/// Block size: 256 elements.
/// Layout: [scales: 16][qs: 64][d: f16][dmin: f16] = 84 bytes
fn dequantize_q2_k(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 16 + 64 + 2 + 2; // scales + qs + d + dmin

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError("Q2_K tensor size not multiple of block size".into()));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = half::f16::from_bits(u16::from_le_bytes(block[80..82].try_into().unwrap())).to_f32();
        let min = half::f16::from_bits(u16::from_le_bytes(block[82..84].try_into().unwrap())).to_f32();

        let mut is = 0;
        let mut q_offset = 0;
        for _n in 0..2 {
            // Two halves of 128 elements each
            let mut shift = 0;
            for _j in 0..4 {
                // 4 groups of 32 elements each
                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    let q = ((qs[q_offset + l] >> shift) & 3) as f32;
                    out.push(dl * q - ml);
                }

                let sc = scales[is];
                is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    let q = ((qs[q_offset + l + 16] >> shift) & 3) as f32;
                    out.push(dl * q - ml);
                }

                shift += 2;
            }
            q_offset += 32;
        }
    }

    Ok(out)
}

/// Q3_K: 3-bit K-quant.
/// Block size: 256 elements.
/// Layout: [hmask: 32][qs: 64][scales: 12][d: f16] = 110 bytes
fn dequantize_q3_k(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 32 + 64 + 12 + 2; // hmask + qs + scales + d

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError("Q3_K tensor size not multiple of block size".into()));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw = &block[96..108];
        let d_all = half::f16::from_bits(u16::from_le_bytes(block[108..110].try_into().unwrap())).to_f32();

        // Unpack scales from 12 bytes into 16 signed scale values
        let kmask1: u32 = 0x03030303;
        let kmask2: u32 = 0x0f0f0f0f;

        let mut aux: [u32; 4] = [
            u32::from_le_bytes(scales_raw[0..4].try_into().unwrap()),
            u32::from_le_bytes(scales_raw[4..8].try_into().unwrap()),
            u32::from_le_bytes(scales_raw[8..12].try_into().unwrap()),
            0,
        ];
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        // Convert to signed scales (each byte is a scale value, subtract 32 for signed)
        let scales: [i8; 16] = [
            aux[0] as i8, (aux[0] >> 8) as i8, (aux[0] >> 16) as i8, (aux[0] >> 24) as i8,
            aux[1] as i8, (aux[1] >> 8) as i8, (aux[1] >> 16) as i8, (aux[1] >> 24) as i8,
            aux[2] as i8, (aux[2] >> 8) as i8, (aux[2] >> 16) as i8, (aux[2] >> 24) as i8,
            aux[3] as i8, (aux[3] >> 8) as i8, (aux[3] >> 16) as i8, (aux[3] >> 24) as i8,
        ];

        let mut is = 0;
        let mut q_offset = 0;
        let mut m: u8 = 1;
        for _n in 0..2 {
            let mut shift = 0;
            for _j in 0..4 {
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let q = ((qs[q_offset + l] >> shift) & 3) as i8;
                    let hm = if hmask[q_offset + l] & m != 0 { 0i8 } else { 4i8 };
                    out.push(dl * (q - hm) as f32);
                }

                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let q = ((qs[q_offset + l + 16] >> shift) & 3) as i8;
                    let hm = if hmask[q_offset + l + 16] & m != 0 { 0i8 } else { 4i8 };
                    out.push(dl * (q - hm) as f32);
                }

                shift += 2;
                m <<= 1;
            }
            q_offset += 32;
        }
    }

    Ok(out)
}

/// Extract scale and min from K-quant scales array.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Q4_K: 4-bit K-quant.
/// Block size: 256 elements.
/// Layout: [d: f16][dmin: f16][scales: 12 bytes][qs: 128 bytes] = 144 bytes
fn dequantize_q4_k(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 2 + 12 + 128;

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError("Q4_K tensor size not multiple of block size".into()));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap())).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut is = 0;
        for _j in 0..4 {
            // 4 groups of 64 elements each
            let (sc, m) = get_scale_min_k4(is, scales);
            is += 1;
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;

            let (sc, m) = get_scale_min_k4(is, scales);
            is += 1;
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            for l in 0..32 {
                out.push(d1 * (qs[l] & 0xF) as f32 - m1);
            }
            for l in 0..32 {
                out.push(d2 * (qs[l] >> 4) as f32 - m2);
            }
        }
    }

    Ok(out)
}

/// Q5_K: 5-bit K-quant.
/// Block size: 256 elements.
/// Layout: [d: f16][dmin: f16][scales: 12][qh: 32][qs: 128] = 176 bytes
fn dequantize_q5_k(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 2 + 2 + 12 + 32 + 128;

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError("Q5_K tensor size not multiple of block size".into()));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let d = half::f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap())).to_f32();
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        let mut is = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _j in 0..4 {
            let (sc, m) = get_scale_min_k4(is, scales);
            is += 1;
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;

            let (sc, m) = get_scale_min_k4(is, scales);
            is += 1;
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;

            for l in 0..32 {
                let high = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                out.push(d1 * ((qs[l] & 0xF) as f32 + high) - m1);
            }
            for l in 0..32 {
                let high = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                out.push(d2 * ((qs[l] >> 4) as f32 + high) - m2);
            }

            u1 <<= 2;
            u2 <<= 2;
        }
    }

    Ok(out)
}

/// Q6_K: 6-bit K-quant.
/// Block size: 256 elements.
/// Layout: [ql: 128][qh: 64][scales: 16][d: f16] = 210 bytes
fn dequantize_q6_k(raw: &[u8]) -> Result<Vec<f32>, GgufError> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 128 + 64 + 16 + 2;

    if raw.len() % BLOCK_SIZE != 0 {
        return Err(GgufError::DecodeError("Q6_K tensor size not multiple of block size".into()));
    }

    let num_blocks = raw.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(num_blocks * QK_K);

    for block in raw.chunks_exact(BLOCK_SIZE) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales_raw = &block[192..208];
        let d = half::f16::from_bits(u16::from_le_bytes(block[208..210].try_into().unwrap())).to_f32();

        let mut ql_offset = 0;
        let mut qh_offset = 0;
        let mut sc_offset = 0;
        for _n in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_offset + l] & 0xF) | ((qh[qh_offset + l] & 0x03) << 4)) as i8 - 32;
                let q2 = ((ql[ql_offset + l + 32] & 0xF) | (((qh[qh_offset + l] >> 2) & 0x03) << 4)) as i8 - 32;
                let q3 = ((ql[ql_offset + l] >> 4) | (((qh[qh_offset + l] >> 4) & 0x03) << 4)) as i8 - 32;
                let q4 = ((ql[ql_offset + l + 32] >> 4) | (((qh[qh_offset + l] >> 6) & 0x03) << 4)) as i8 - 32;

                let s0 = scales_raw[sc_offset + is * 2 + 0] as i8;
                let s2 = scales_raw[sc_offset + is * 2 + 2] as i8;
                let s4 = scales_raw[sc_offset + is * 2 + 4] as i8;
                let s6 = scales_raw[sc_offset + is * 2 + 6] as i8;

                out.push(d * s0 as f32 * q1 as f32);
                out.push(d * s2 as f32 * q2 as f32);
                out.push(d * s4 as f32 * q3 as f32);
                out.push(d * s6 as f32 * q4 as f32);
            }
            ql_offset += 64;
            qh_offset += 32;
            sc_offset += 8;
        }
    }

    Ok(out)
}

// ─── GGUF Reader ─────────────────────────────────────────────────────────────

/// A GGUF file reader that memory-maps the file for efficient access.
pub struct GgufReader {
    /// Memory-mapped file data.
    data: memmap2::Mmap,
    /// GGUF version.
    version: u32,
    /// Number of tensors.
    tensor_count: i64,
    /// Number of KV pairs.
    metadata_count: i64,
    /// Metadata key-value pairs.
    kv_pairs: Vec<(String, GgufValue)>,
    /// Tensor info entries.
    tensors: Vec<TensorInfo>,
    /// Alignment (from general.alignment or default).
    alignment: usize,
    /// Offset where tensor data begins.
    data_offset: usize,
}

// Added helper methods for convenience
impl GgufReader {
    /// Get a usize metadata value by key (expects u32 stored).
    ///
    /// # Errors
    ///
    /// Returns a `GgufError::DecodeError` if the key is missing or has an
    /// unexpected type.
    pub fn get_usize(&self, key: &str) -> GgufResult<usize> {
        match self.get_kv(key) {
            Some(GgufValue::U32(v)) => Ok(*v as usize),
            Some(GgufValue::U64(v)) => Ok(usize::try_from(*v).map_err(|e| GgufError::DecodeError(e.to_string()))?),
            Some(other) => Err(GgufError::DecodeError(format!("metadata key '{key}' has unexpected type: {other:?}"))),
            None => Err(GgufError::DecodeError(format!("metadata key '{key}' not found"))),
        }
    }

    /// Get a usize metadata value, trying multiple keys in order.
    /// Returns the first key that exists and has a valid type.
    ///
    /// # Errors
    ///
    /// Returns a `GgufError::DecodeError` if none of the keys are found or
    /// all have unexpected types.
    pub fn get_usize_any(&self, keys: &[&str]) -> GgufResult<usize> {
        for &key in keys {
            if let Some(val) = self.get_kv(key) {
                return match val {
                    GgufValue::U32(v) => Ok(*v as usize),
                    GgufValue::U64(v) => Ok(usize::try_from(*v).map_err(|e| GgufError::DecodeError(e.to_string()))?),
                    other => Err(GgufError::DecodeError(format!("metadata key '{key}' has unexpected type: {other:?}"))),
                };
            }
        }
        Err(GgufError::DecodeError(format!("none of the metadata keys found: {:?}", keys)))
    }

    /// Get a string metadata value by key.
    ///
    /// # Errors
    ///
    /// Returns a `GgufError::DecodeError` if the key is missing or has an
    /// unexpected type.
    pub fn get_string(&self, key: &str) -> GgufResult<String> {
        match self.get_kv(key) {
            Some(GgufValue::Str(s)) => Ok(s.clone()),
            Some(other) => Err(GgufError::DecodeError(format!("metadata key '{key}' has unexpected type: {other:?}"))),
            None => Err(GgufError::DecodeError(format!("metadata key '{key}' not found"))),
        }
    }

    /// Get an array of strings metadata value by key.
    ///
    /// # Errors
    ///
    /// Returns a `GgufError::DecodeError` if the key is missing, has an
    /// unexpected type, or the array contains non-string elements.
    pub fn get_string_array(&self, key: &str) -> GgufResult<Vec<String>> {
        match self.get_kv(key) {
            Some(GgufValue::Array { elem_type, data }) => {
                if *elem_type != GgufType::String {
                    return Err(GgufError::DecodeError(format!("metadata key '{key}' expected string array, got {elem_type:?}")));
                }
                let mut result = Vec::with_capacity(data.len());
                for val in data {
                    if let GgufValue::Str(s) = val {
                        result.push(s.clone());
                    } else {
                        return Err(GgufError::DecodeError(format!("metadata key '{key}' contains non-string element")));
                    }
                }
                Ok(result)
            }
            Some(other) => Err(GgufError::DecodeError(format!("metadata key '{key}' has unexpected type: {other:?}"))),
            None => Err(GgufError::DecodeError(format!("metadata key '{key}' not found"))),
        }
    }

    /// Get an array of f32 metadata value by key.
    ///
    /// # Errors
    ///
    /// Returns a `GgufError::DecodeError` if the key is missing, has an
    /// unexpected type, or the array contains non-f32 elements.
    pub fn get_f32_array(&self, key: &str) -> GgufResult<Vec<f32>> {
        match self.get_kv(key) {
            Some(GgufValue::Array { elem_type, data }) => {
                if *elem_type != GgufType::Float32 {
                    return Err(GgufError::DecodeError(format!("metadata key '{key}' expected f32 array, got {elem_type:?}")));
                }
                let mut result = Vec::with_capacity(data.len());
                for val in data {
                    if let GgufValue::F32(v) = val {
                        result.push(*v);
                    } else {
                        return Err(GgufError::DecodeError(format!("metadata key '{key}' contains non-f32 element")));
                    }
                }
                Ok(result)
            }
            Some(other) => Err(GgufError::DecodeError(format!("metadata key '{key}' has unexpected type: {other:?}"))),
            None => Err(GgufError::DecodeError(format!("metadata key '{key}' not found"))),
        }
    }

    /// Get an array of i32 metadata value by key.
    ///
    /// # Errors
    ///
    /// Returns a `GgufError::DecodeError` if the key is missing, has an
    /// unexpected type, or the array contains non-i32 elements.
    pub fn get_i32_array(&self, key: &str) -> GgufResult<Vec<i32>> {
        match self.get_kv(key) {
            Some(GgufValue::Array { elem_type, data }) => {
                if *elem_type != GgufType::Int32 {
                    return Err(GgufError::DecodeError(format!("metadata key '{key}' expected i32 array, got {elem_type:?}")));
                }
                let mut result = Vec::with_capacity(data.len());
                for val in data {
                    if let GgufValue::I32(v) = val {
                        result.push(*v);
                    } else {
                        return Err(GgufError::DecodeError(format!("metadata key '{key}' contains non-i32 element")));
                    }
                }
                Ok(result)
            }
            Some(other) => Err(GgufError::DecodeError(format!("metadata key '{key}' has unexpected type: {other:?}"))),
            None => Err(GgufError::DecodeError(format!("metadata key '{key}' not found"))),
        }
    }

    /// Load raw tensor bytes for a given `TensorInfo`.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] if the tensor data cannot be read.
    pub fn load_tensor_raw(&self, info: &TensorInfo) -> GgufResult<&[u8]> {
        self.read_tensor_data(info)
    }

    // Existing methods follow...

    /// Open a GGUF file from the given path.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] if the file cannot be opened or is not a valid GGUF file.
    pub fn from_file(path: impl AsRef<Path>) -> GgufResult<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    /// Parse a GGUF file from a memory-mapped region.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] if the data is not a valid GGUF file.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap
    )]
    pub fn from_mmap(mmap: memmap2::Mmap) -> GgufResult<Self> {
        let mut reader = CursorReader::new(&mmap);

        // 1. Magic
        let magic = reader.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic);
        }

        // 2. Version
        let version = reader.read_u32()?;
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // 3. Tensor count
        let tensor_count = reader.read_i64()?;

        // 4. KV pair count
        let metadata_count = reader.read_i64()?;

        // 5. KV pairs
        let mut kv_pairs = Vec::with_capacity(metadata_count as usize);
        for _ in 0..metadata_count {
            let key = reader.read_string()?;
            let type_raw = reader.read_i32()?;
            let gguf_type = GgufType::from_i32(type_raw)?;
            let value = reader.read_value(gguf_type)?;
            kv_pairs.push((key, value));
        }

        // Determine alignment from metadata
        let mut alignment = GGUF_DEFAULT_ALIGNMENT;
        for (key, value) in &kv_pairs {
            if key == "general.alignment" {
                if let GgufValue::U32(v) = value {
                    alignment = *v as usize;
                }
                break;
            }
        }

        // 6. Tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let n_dims = reader.read_u32()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(reader.read_i64()?);
            }
            let dtype_raw = reader.read_i32()?;
            let dtype = GgmlType::from_i32(dtype_raw)?;
            let offset = reader.read_u64()?;
            tensors.push(TensorInfo {
                name,
                n_dims,
                shape,
                dtype,
                offset,
            });
        }

        // 7. Data offset (current position, aligned)
        let data_offset = reader.position();
        let aligned_offset = align_up(data_offset, alignment);

        Ok(Self {
            data: mmap,
            version,
            tensor_count,
            metadata_count,
            kv_pairs,
            tensors,
            alignment,
            data_offset: aligned_offset,
        })
    }

    /// Returns the GGUF version.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns the number of tensors.
    #[must_use]
    pub fn tensor_count(&self) -> i64 {
        self.tensor_count
    }

    /// Returns the number of metadata KV pairs.
    #[must_use]
    pub fn metadata_count(&self) -> i64 {
        self.metadata_count
    }

    /// Returns the alignment used for tensor data.
    #[must_use]
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Returns the offset where tensor data begins.
    #[must_use]
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// Get a metadata value by key.
    ///
    /// Returns `None` if the key is not found.
    #[must_use]
    pub fn get_kv(&self, key: &str) -> Option<&GgufValue> {
        self.kv_pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Get a metadata key-value pair by index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of range.
    pub fn kv_pair(&self, index: usize) -> GgufResult<(&str, &GgufValue)> {
        if index >= self.kv_pairs.len() {
            return Err(GgufError::KvIndexOutOfRange(index, self.kv_pairs.len()));
        }
        let (ref k, ref v) = self.kv_pairs[index];
        Ok((k, v))
    }

    /// Get tensor info by index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of range.
    pub fn tensor_info(&self, index: usize) -> GgufResult<&TensorInfo> {
        if index >= self.tensors.len() {
            return Err(GgufError::TensorIndexOutOfRange(index, self.tensors.len()));
        }
        Ok(&self.tensors[index])
    }

    /// Find a tensor by name.
    ///
    /// Returns `None` if not found.
    #[must_use]
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Returns a slice of all tensor info entries.
    #[must_use]
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Returns a slice of all KV pairs.
    #[must_use]
    pub fn kv_pairs(&self) -> &[(String, GgufValue)] {
        &self.kv_pairs
    }

    /// Get a reference to the raw mmap data.
    #[must_use]
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }

    /// Read tensor data bytes from the memory-mapped file.
    ///
    /// The data is located at `data_offset + tensor.offset`.
    ///
    /// # Errors
    ///
    /// Returns an error if the offset is out of bounds.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn read_tensor_data(&self, tensor: &TensorInfo) -> GgufResult<&[u8]> {
        let start = self.data_offset + tensor.offset as usize;
        if start > self.data.len() {
            return Err(GgufError::DecodeError(format!(
                "tensor {} offset {} out of bounds (file size {})",
                tensor.name,
                tensor.offset,
                self.data.len()
            )));
        }

        // Calculate tensor size from shape and dtype
        let element_count: usize = tensor.shape.iter().map(|&d| d as usize).product();
        let byte_size = match tensor.dtype {
            GgmlType::F32 | GgmlType::I32 => element_count * 4,
            GgmlType::F16 | GgmlType::I16 | GgmlType::Bf16 => element_count * 2,
            GgmlType::F64 | GgmlType::I64 => element_count * 8,
            GgmlType::I8 | GgmlType::Q8_0 | GgmlType::Q8_1 | GgmlType::Q8_K => element_count,
            GgmlType::Q4_0 | GgmlType::Q4_1 => element_count / 2,
            GgmlType::Q5_0 | GgmlType::Q5_1 => (element_count / 2) + (element_count / 32) * 2,
            GgmlType::Q2_K | GgmlType::Q3_K => {
                element_count / 4 + element_count / 64 + element_count / 64
            }
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => {
                element_count / 2 + element_count / 64 + element_count / 64
            }
            GgmlType::Iq2Xxs
            | GgmlType::Iq2Xs
            | GgmlType::Iq3Xxs
            | GgmlType::Iq1S
            | GgmlType::Iq4Nl
            | GgmlType::Iq3S
            | GgmlType::Iq2S
            | GgmlType::Iq4Xs
            | GgmlType::Iq1M
            | GgmlType::Tq1_0
            | GgmlType::Tq2_0
            | GgmlType::Mxfp4
            | GgmlType::Nvfp4
            | GgmlType::Q1_0 => {
                return Err(GgufError::DecodeError(format!(
                    "unsupported quantized dtype for direct read: {:?}",
                    tensor.dtype
                )));
            }
        };

        let end = start + byte_size;
        if end > self.data.len() {
            return Err(GgufError::DecodeError(format!(
                "tensor {} data extends beyond file (need {}, have {})",
                tensor.name,
                end,
                self.data.len()
            )));
        }

        Ok(&self.data[start..end])
    }
}

// ─── Binary Reader ───────────────────────────────────────────────────────────

/// Little-endian binary reader over a byte slice.
struct CursorReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> CursorReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn ensure(&self, n: usize) -> GgufResult<()> {
        if self.pos + n > self.data.len() {
            Err(GgufError::UnexpectedEof)
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> GgufResult<u8> {
        self.ensure(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> GgufResult<i8> {
        self.ensure(1)?;
        let v = i8::from_ne_bytes([self.data[self.pos]]);
        self.pos += 1;
        Ok(v)
    }

    fn read_u16(&mut self) -> GgufResult<u16> {
        self.ensure(2)?;
        let v = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> GgufResult<i16> {
        self.ensure(2)?;
        let v = i16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn read_u32(&mut self) -> GgufResult<u32> {
        self.ensure(4)?;
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> GgufResult<i32> {
        self.ensure(4)?;
        let v = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> GgufResult<u64> {
        self.ensure(8)?;
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> GgufResult<i64> {
        self.ensure(8)?;
        let v = i64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_f32(&mut self) -> GgufResult<f32> {
        self.ensure(4)?;
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_f64(&mut self) -> GgufResult<f64> {
        self.ensure(8)?;
        let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn read_string(&mut self) -> GgufResult<String> {
        let len = self.read_u64()? as usize;
        self.ensure(len)?;
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|e| GgufError::DecodeError(format!("invalid UTF-8 in string: {e}")))?;
        let s = s.to_string();
        self.pos += len;
        Ok(s)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn read_value(&mut self, gguf_type: GgufType) -> GgufResult<GgufValue> {
        match gguf_type {
            GgufType::Uint8 => Ok(GgufValue::U8(self.read_u8()?)),
            GgufType::Int8 => Ok(GgufValue::I8(self.read_i8()?)),
            GgufType::Uint16 => Ok(GgufValue::U16(self.read_u16()?)),
            GgufType::Int16 => Ok(GgufValue::I16(self.read_i16()?)),
            GgufType::Uint32 => Ok(GgufValue::U32(self.read_u32()?)),
            GgufType::Int32 => Ok(GgufValue::I32(self.read_i32()?)),
            GgufType::Float32 => Ok(GgufValue::F32(self.read_f32()?)),
            GgufType::Bool => {
                let v = self.read_i8()?;
                Ok(GgufValue::Bool(v != 0))
            }
            GgufType::String => Ok(GgufValue::Str(self.read_string()?)),
            GgufType::Uint64 => Ok(GgufValue::U64(self.read_u64()?)),
            GgufType::Int64 => Ok(GgufValue::I64(self.read_i64()?)),
            GgufType::Float64 => Ok(GgufValue::F64(self.read_f64()?)),
            GgufType::Array => {
                let elem_type_raw = self.read_i32()?;
                let elem_type = GgufType::from_i32(elem_type_raw)?;
                let n = self.read_u64()? as usize;
                let mut data = Vec::with_capacity(n);
                for _ in 0..n {
                    data.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array { elem_type, data })
            }
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Align `val` up to the next multiple of `alignment`.
fn align_up(val: usize, alignment: usize) -> usize {
    (val + alignment - 1) & !(alignment - 1)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_magic_constant_should_be_correct() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
        assert_eq!(
            core::str::from_utf8(&GGUF_MAGIC.to_le_bytes()).unwrap(),
            "GGUF"
        );
    }

    #[test]
    fn gguf_version_should_be_three() {
        assert_eq!(GGUF_VERSION, 3);
    }

    #[test]
    fn gguf_type_size_should_be_correct() {
        assert_eq!(GgufType::Uint8.size_of(), 1);
        assert_eq!(GgufType::Float32.size_of(), 4);
        assert_eq!(GgufType::Float64.size_of(), 8);
        assert_eq!(GgufType::String.size_of(), 0);
        assert_eq!(GgufType::Array.size_of(), 0);
    }

    #[test]
    fn gguf_type_from_i32_should_be_correct() {
        assert_eq!(GgufType::from_i32(0).unwrap(), GgufType::Uint8);
        assert_eq!(GgufType::from_i32(6).unwrap(), GgufType::Float32);
        assert_eq!(GgufType::from_i32(9).unwrap(), GgufType::Array);
        assert_eq!(GgufType::from_i32(12).unwrap(), GgufType::Float64);
        assert!(GgufType::from_i32(13).is_err());
    }

    #[test]
    fn ggml_type_from_i32_should_be_correct() {
        assert_eq!(GgmlType::from_i32(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_i32(1).unwrap(), GgmlType::F16);
        assert_eq!(GgmlType::from_i32(2).unwrap(), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_i32(30).unwrap(), GgmlType::Bf16);
        assert!(GgmlType::from_i32(31).is_err()); // removed type
    }

    #[test]
    fn align_up_should_round_correctly() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(63, 32), 64);
        assert_eq!(align_up(64, 32), 64);
    }

    #[test]
    fn from_file_should_return_error_for_missing_file() {
        let result = GgufReader::from_file("/nonexistent/path/file.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn from_mmap_should_reject_invalid_magic() {
        let data = [0u8; 16];
        let mut reader = CursorReader::new(&data);
        let magic = reader.read_u32().unwrap();
        assert_ne!(magic, GGUF_MAGIC);
    }

    #[test]
    fn cursor_reader_should_read_little_endian() {
        // 0x00000001 in little-endian
        let data = [0x01, 0x00, 0x00, 0x00];
        let mut reader = CursorReader::new(&data);
        assert_eq!(reader.read_u32().unwrap(), 1);
    }

    #[test]
    fn cursor_reader_should_read_string() {
        // length=5 + "hello"
        let data: Vec<u8> = [
            0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // len=5
            b'h', b'e', b'l', b'l', b'o',
        ]
        .to_vec();
        let mut reader = CursorReader::new(&data);
        assert_eq!(reader.read_string().unwrap(), "hello");
    }

    #[test]
    fn should_parse_minimal_gguf_file() {
        // Build a minimal valid GGUF v3 file in memory:
        // magic(4) + version(4) + tensor_count(8) + kv_count(8) = 24 bytes header
        // Then 0 KV pairs, 0 tensors
        let mut data = Vec::new();

        // Magic
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        // Tensor count = 0
        data.extend_from_slice(&0i64.to_le_bytes());
        // KV count = 0
        data.extend_from_slice(&0i64.to_le_bytes());

        // Test via CursorReader
        let mut reader = CursorReader::new(&data);
        assert_eq!(reader.read_u32().unwrap(), GGUF_MAGIC);
        assert_eq!(reader.read_u32().unwrap(), GGUF_VERSION);
        assert_eq!(reader.read_i64().unwrap(), 0);
        assert_eq!(reader.read_i64().unwrap(), 0);
    }

    #[test]
    fn should_parse_gguf_with_kv_pair() {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&0i64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&1i64.to_le_bytes()); // 1 KV pair

        // KV pair: key="general.architecture", value="llama" (string)
        let key = "general.architecture";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&(GgufType::String as i32).to_le_bytes());
        let val = "llama";
        data.extend_from_slice(&(val.len() as u64).to_le_bytes());
        data.extend_from_slice(val.as_bytes());

        let mut reader = CursorReader::new(&data);
        assert_eq!(reader.read_u32().unwrap(), GGUF_MAGIC);
        assert_eq!(reader.read_u32().unwrap(), GGUF_VERSION);
        assert_eq!(reader.read_i64().unwrap(), 0);
        assert_eq!(reader.read_i64().unwrap(), 1);

        let gguf_key = reader.read_string().unwrap();
        assert_eq!(gguf_key, "general.architecture");

        let type_raw = reader.read_i32().unwrap();
        assert_eq!(GgufType::from_i32(type_raw).unwrap(), GgufType::String);

        let gguf_val = reader.read_string().unwrap();
        assert_eq!(gguf_val, "llama");
    }

    #[test]
    fn should_parse_gguf_with_tensor_info() {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&1i64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&0i64.to_le_bytes()); // 0 KV pairs

        // Tensor info
        let name = "output.weight";
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
        data.extend_from_slice(&256i64.to_le_bytes()); // dim 0
        data.extend_from_slice(&4096i64.to_le_bytes()); // dim 1
        data.extend_from_slice(&(GgmlType::F32 as i32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        let mut reader = CursorReader::new(&data);
        assert_eq!(reader.read_u32().unwrap(), GGUF_MAGIC);
        assert_eq!(reader.read_u32().unwrap(), GGUF_VERSION);
        assert_eq!(reader.read_i64().unwrap(), 1); // tensor count
        assert_eq!(reader.read_i64().unwrap(), 0); // kv count

        let t_name = reader.read_string().unwrap();
        assert_eq!(t_name, "output.weight");

        let n_dims = reader.read_u32().unwrap();
        assert_eq!(n_dims, 2);

        assert_eq!(reader.read_i64().unwrap(), 256);
        assert_eq!(reader.read_i64().unwrap(), 4096);

        let dtype_raw = reader.read_i32().unwrap();
        assert_eq!(GgmlType::from_i32(dtype_raw).unwrap(), GgmlType::F32);

        let offset = reader.read_u64().unwrap();
        assert_eq!(offset, 0);
    }

    #[test]
    fn should_parse_realistic_llama_gguf() {
        // Build a realistic GGUF file with llama architecture metadata
        // and tensor info, then parse it with GgufReader via mmap
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        data.extend_from_slice(&2i64.to_le_bytes()); // 2 tensors
        data.extend_from_slice(&5i64.to_le_bytes()); // 5 KV pairs

        // KV pairs
        fn write_kv_string(data: &mut Vec<u8>, key: &str, val: &str) {
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            data.extend_from_slice(&(GgufType::String as i32).to_le_bytes());
            data.extend_from_slice(&(val.len() as u64).to_le_bytes());
            data.extend_from_slice(val.as_bytes());
        }

        fn write_kv_u32(data: &mut Vec<u8>, key: &str, val: u32) {
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            data.extend_from_slice(&(GgufType::Uint32 as i32).to_le_bytes());
            data.extend_from_slice(&val.to_le_bytes());
        }

        fn write_kv_f32(data: &mut Vec<u8>, key: &str, val: f32) {
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            data.extend_from_slice(&(GgufType::Float32 as i32).to_le_bytes());
            data.extend_from_slice(&val.to_le_bytes());
        }

        write_kv_string(&mut data, "general.architecture", "llama");
        write_kv_u32(&mut data, "llama.embedding_length", 256);
        write_kv_u32(&mut data, "llama.attention.head_count", 8);
        write_kv_u32(&mut data, "llama.block_count", 4);
        write_kv_f32(&mut data, "llama.attention.layer_norm_rms_epsilon", 1e-5);

        // Tensor info (comes after KV pairs)
        let tensor1_name = "token_embd.weight";
        data.extend_from_slice(&(tensor1_name.len() as u64).to_le_bytes());
        data.extend_from_slice(tensor1_name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // 2 dims
        data.extend_from_slice(&256i64.to_le_bytes());
        data.extend_from_slice(&4096i64.to_le_bytes());
        data.extend_from_slice(&(GgmlType::F32 as i32).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset 0

        let tensor2_name = "output.weight";
        data.extend_from_slice(&(tensor2_name.len() as u64).to_le_bytes());
        data.extend_from_slice(tensor2_name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&256i64.to_le_bytes());
        data.extend_from_slice(&4096i64.to_le_bytes());
        data.extend_from_slice(&(GgmlType::F16 as i32).to_le_bytes());
        // offset = size of tensor 1 data (256 * 4096 * 4 bytes), aligned to 32
        let t1_size = 256 * 4096 * 4;
        let t1_aligned = (t1_size + 31) & !31;
        data.extend_from_slice(&(t1_aligned as u64).to_le_bytes());

        // Pad to alignment for tensor data section
        let tensor_data_start = data.len();
        let aligned_data_start = (tensor_data_start + 31) & !31;
        while data.len() < aligned_data_start {
            data.push(0);
        }

        // Write dummy tensor data
        // Tensor 1: 256 * 4096 * 4 bytes (F32)
        let t1_size = 256 * 4096 * 4;
        data.resize(data.len() + t1_size, 0);

        // Tensor 2: 256 * 4096 * 2 bytes (F16)
        let t2_size = 256 * 4096 * 2;
        data.resize(data.len() + t2_size, 0);

        // Parse with GgufReader
        let mut file = std::fs::File::create("/tmp/test_llama.gguf").unwrap();
        use std::io::Write;
        file.write_all(&data).unwrap();
        drop(file);

        let reader = GgufReader::from_file("/tmp/test_llama.gguf").unwrap();

        // Verify metadata
        assert_eq!(reader.tensor_count(), 2);
        assert_eq!(reader.metadata_count(), 5);

        let arch = reader.get_kv("general.architecture").unwrap();
        if let GgufValue::Str(s) = arch {
            assert_eq!(s, "llama");
        } else {
            panic!("expected string");
        }

        let embd = reader.get_kv("llama.embedding_length").unwrap();
        if let GgufValue::U32(v) = embd {
            assert_eq!(*v, 256);
        } else {
            panic!("expected u32");
        }

        // Verify tensors
        let t1 = reader.find_tensor("token_embd.weight").unwrap();
        assert_eq!(t1.shape, vec![256, 4096]);
        assert_eq!(t1.dtype, GgmlType::F32);

        let t2 = reader.find_tensor("output.weight").unwrap();
        assert_eq!(t2.shape, vec![256, 4096]);
        assert_eq!(t2.dtype, GgmlType::F16);

        // Clean up
        let _ = std::fs::remove_file("/tmp/test_llama.gguf");
    }
}
