//! Core tensor library and computation graph.
//!
//! This crate provides the fundamental data structures for tensor operations,
//! including tensor storage, shapes, data types, and computation graphs.
//!
//! # Example
//!
//! ```
//! use ggml::Tensor;
//! use ggml::DType;
//!
//! let tensor = Tensor::new(DType::F32, &[2, 3]);
//! assert_eq!(tensor.shape().len(), 2);
//! ```

#![deny(missing_docs)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]

use thiserror::Error;

/// Errors that can occur during tensor operations.
#[derive(Debug, Error)]
pub enum TensorError {
    /// The tensor shape is invalid for the given operation.
    #[error("invalid shape: {0}")]
    InvalidShape(String),

    /// The data type is not supported.
    #[error("unsupported dtype: {0:?}")]
    UnsupportedDType(DType),

    /// The tensor size exceeds available memory.
    #[error("tensor too large: {0} elements")]
    TensorTooLarge(usize),
}

/// Result type alias for tensor operations.
pub type TensorResult<T> = Result<T, TensorError>;

/// Data types supported by the tensor library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point (IEEE 754-2008 binary16).
    F16,
    /// 16-bit brain floating point.
    BF16,
    /// 8-bit integer.
    I8,
    /// 16-bit integer.
    I16,
    /// 32-bit integer.
    I32,
    /// 64-bit integer.
    I64,
    /// 4-bit quantized (`Q4_0`).
    Q4_0,
    /// 4-bit quantized (`Q4_1`).
    Q4_1,
    /// 8-bit quantized (`Q8_0`).
    Q8_0,
}

impl DType {
    /// Returns the size in bytes of a single element of this type.
    #[must_use]
    pub fn size_of(self) -> f64 {
        match self {
            DType::F32 | DType::I32 => 4.0,
            DType::F16 | DType::BF16 | DType::I16 => 2.0,
            DType::I64 => 8.0,
            DType::I8 | DType::Q8_0 => 1.0,
            DType::Q4_0 | DType::Q4_1 => 0.5,
        }
    }
}

/// A multi-dimensional tensor with owned data.
#[derive(Debug, Clone)]
pub struct Tensor {
    dtype: DType,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl Tensor {
    /// Create a new tensor with the given dtype and shape, zero-initialized.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn new(dtype: DType, shape: &[usize]) -> Self {
        let element_count: usize = shape.iter().product();
        let byte_size = (element_count as f64 * dtype.size_of()).ceil() as usize;

        Self {
            dtype,
            shape: shape.to_vec(),
            data: vec![0; byte_size],
        }
    }

    /// Returns the data type of this tensor.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the shape of this tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns the size of the data in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Returns a reference to the raw byte data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns a mutable reference to the raw byte data.
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Create a tensor from an f32 slice with the given shape.
    ///
    /// # Panics
    ///
    /// Panics if the slice length doesn't match the shape.
    #[must_use]
    pub fn from_f32(shape: &[usize], data: &[f32]) -> Self {
        let element_count: usize = shape.iter().product();
        assert_eq!(data.len(), element_count, "data length must match shape");
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        Self {
            dtype: DType::F32,
            shape: shape.to_vec(),
            data: bytes.to_vec(),
        }
    }
}

/// A computation graph representing a sequence of tensor operations.
pub struct ComputeGraph {
    nodes: Vec<GraphNode>,
    finalized: bool,
}

/// A single operation node in the computation graph.
#[derive(Debug)]
#[allow(dead_code)]
pub struct GraphNode {
    name: String,
    op: GraphOp,
}

/// Types of operations in the computation graph.
#[derive(Debug)]
pub enum GraphOp {
    /// Element-wise addition.
    Add,
    /// Element-wise multiplication.
    Mul,
    /// Matrix multiplication.
    MatMul,
    /// `ReLU` activation.
    Relu,
    /// Softmax activation.
    Softmax,
    /// No-op (placeholder).
    None,
}

impl ComputeGraph {
    /// Create a new empty computation graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            finalized: false,
        }
    }

    /// Add a new node to the computation graph.
    ///
    /// # Panics
    ///
    /// Panics if the graph has already been finalized.
    pub fn add_node(&mut self, name: impl Into<String>, op: GraphOp) {
        assert!(!self.finalized, "cannot modify a finalized graph");
        self.nodes.push(GraphNode {
            name: name.into(),
            op,
        });
    }

    /// Finalize the graph, preventing further modifications.
    pub fn finalize(&mut self) {
        self.finalized = true;
    }

    /// Returns the number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns whether the graph has been finalized.
    #[must_use]
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tensor_should_have_correct_shape() {
        let tensor = Tensor::new(DType::F32, &[2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.ndim(), 2);
    }

    #[test]
    fn new_tensor_should_be_zero_initialized() {
        let tensor = Tensor::new(DType::F32, &[2, 2]);
        assert!(tensor.data().iter().all(|&b| b == 0));
    }

    #[test]
    fn dtype_size_should_be_correct() {
        assert_eq!(DType::F32.size_of(), 4.0);
        assert_eq!(DType::F16.size_of(), 2.0);
        assert_eq!(DType::Q4_0.size_of(), 0.5);
    }

    #[test]
    fn compute_graph_should_track_node_count() {
        let mut graph = ComputeGraph::new();
        assert_eq!(graph.node_count(), 0);

        graph.add_node("add1", GraphOp::Add);
        graph.add_node("mul1", GraphOp::Mul);
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn finalize_should_prevent_modifications() {
        let mut graph = ComputeGraph::new();
        graph.finalize();
        assert!(graph.is_finalized());
    }

    #[test]
    #[should_panic(expected = "cannot modify a finalized graph")]
    fn add_node_should_panic_after_finalize() {
        let mut graph = ComputeGraph::new();
        graph.finalize();
        graph.add_node("bad", GraphOp::None);
    }
}
