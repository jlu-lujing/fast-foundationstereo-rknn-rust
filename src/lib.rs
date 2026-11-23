//! Fast-FoundationStereo inference on Rockchip RK3588.
//!
//! The RKNN runtime is loaded dynamically. ONNX Runtime is also loaded at
//! runtime by the `ort` backend, so binaries can be built off-device.

pub mod cpu;
pub mod gwc;
pub mod pipeline;
pub mod rknn;
pub mod tensor;

mod error;

pub use error::{Error, Result};
pub use pipeline::{FoundationStereo, ModelArtifacts, PipelineOptions, PipelineTrace};
pub use rknn::{NpuCore, RknnModel, RknnOptions, TensorInfo};
pub use tensor::Tensor;
