//! High-performance Fast-FoundationStereo inference on Rockchip RK3588.
//!
//! The complete stereo network runs in one fixed-shape RKNN graph. The RKNN
//! runtime is loaded dynamically, so the crate can be compiled on a host
//! without the Rockchip SDK.

pub mod rknn;
pub mod stereo;
pub mod tensor;

mod error;

pub use error::{Error, Result};
pub use rknn::{NpuCore, RknnModel, RknnOptions, TensorInfo};
pub use stereo::{F16Image, FoundationStereo};
pub use tensor::Tensor;
