//! Fast-FoundationStereo inference on Rockchip RK3588.
//!
//! The RKNN runtime is loaded dynamically. ONNX Runtime is also loaded at
//! runtime by the `ort` backend, so binaries can be built off-device.

#[cfg(feature = "ort")]
pub mod cpu;
pub mod gwc;
#[cfg(feature = "ort")]
pub mod pipeline;
pub mod rknn;
pub mod single_graph;
pub mod tensor;

mod error;

pub use error::{Error, Result};
#[cfg(feature = "ort")]
pub use pipeline::{FoundationStereo, ModelArtifacts, PipelineOptions, PipelineTrace};
pub use rknn::{NpuCore, RknnModel, RknnOptions, TensorInfo};
pub use single_graph::{F16Image, SingleGraphStereo};
pub use tensor::Tensor;
