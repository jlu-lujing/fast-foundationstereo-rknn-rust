//! Whole-network Fast-FoundationStereo inference in one RKNN graph.

use std::path::Path;

use half::f16;

use crate::{
    Error, Result, Tensor,
    rknn::{RknnModel, RknnOptions},
};

/// RGB image in contiguous NHWC byte order, stored as little-endian FP16 words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F16Image {
    pub height: usize,
    pub width: usize,
    pub data: Vec<u16>,
}

impl F16Image {
    /// Pack an RGB `[1,3,H,W]` float tensor to RKNN-ready NHWC FP16.
    pub fn from_nchw(value: &Tensor) -> Result<Self> {
        let [1, channels, height, width] = value.shape.as_slice() else {
            return Err(Error::InvalidTensor(format!(
                "image must be NCHW [1,C,H,W], got {:?}",
                value.shape
            )));
        };
        if *channels != 3 {
            return Err(Error::InvalidTensor(format!(
                "image must have three RGB channels, got {channels}"
            )));
        }
        let plane = height * width;
        let mut data = Vec::with_capacity(plane * 3);
        for pixel in 0..plane {
            for channel in 0..3 {
                let value = value.data[channel * plane + pixel];
                if !value.is_finite() {
                    return Err(Error::InvalidTensor(format!(
                        "image contains non-finite RGB value at pixel {pixel}, channel {channel}"
                    )));
                }
                data.push(f16::from_f32(value).to_bits());
            }
        }
        Ok(Self {
            height: *height,
            width: *width,
            data,
        })
    }

    /// Pack contiguous RGB8 pixels into NHWC FP16.
    pub fn from_rgb8(height: usize, width: usize, rgb: &[u8]) -> Result<Self> {
        let mut image = Self::zeroed(height, width)?;
        image.copy_from_rgb8(rgb)?;
        Ok(image)
    }

    fn zeroed(height: usize, width: usize) -> Result<Self> {
        let expected = height
            .checked_mul(width)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| Error::InvalidTensor("RGB image dimensions overflow usize".into()))?;
        Ok(Self {
            height,
            width,
            data: vec![0; expected],
        })
    }

    fn copy_from_rgb8(&mut self, rgb: &[u8]) -> Result<()> {
        let expected = self.expected_len();
        if rgb.len() != expected {
            return Err(Error::InvalidTensor(format!(
                "RGB image has {} bytes, expected {expected} for {}x{}",
                rgb.len(),
                self.width,
                self.height
            )));
        }
        for (destination, value) in self.data.iter_mut().zip(rgb) {
            *destination = f16::from_f32(*value as f32).to_bits();
        }
        Ok(())
    }

    fn copy_from_nchw(&mut self, value: &Tensor) -> Result<()> {
        let [1, channels, height, width] = value.shape.as_slice() else {
            return Err(Error::InvalidTensor(format!(
                "image must be NCHW [1,C,H,W], got {:?}",
                value.shape
            )));
        };
        if *channels != 3 || *height != self.height || *width != self.width {
            return Err(Error::InvalidTensor(format!(
                "image is C={channels}, {}x{}, buffer expects C=3, {}x{}",
                width, height, self.width, self.height
            )));
        }
        let plane = height * width;
        for pixel in 0..plane {
            for channel in 0..3 {
                let value = value.data[channel * plane + pixel];
                if !value.is_finite() {
                    return Err(Error::InvalidTensor(format!(
                        "image contains non-finite RGB value at pixel {pixel}, channel {channel}"
                    )));
                }
                self.data[pixel * 3 + channel] = f16::from_f32(value).to_bits();
            }
        }
        Ok(())
    }

    fn expected_len(&self) -> usize {
        self.height * self.width * 3
    }
}

/// Fast-FoundationStereo running as one fixed-shape RKNN graph.
///
/// The model contract is two FP16 NHWC RGB tensors with values in `0..=255`
/// and one FP32 disparity tensor. Model initialization selects all three
/// RK3588 NPU cores by default.
pub struct FoundationStereo {
    model: RknnModel,
    height: usize,
    width: usize,
    left_buffer: F16Image,
    right_buffer: F16Image,
}

impl FoundationStereo {
    /// Load with the default RKNN library lookup and all three NPU cores.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_options(path, RknnOptions::default())
    }

    /// Load with an explicit RKNN runtime path or core selection.
    pub fn load_with_options(path: impl AsRef<Path>, options: RknnOptions) -> Result<Self> {
        let model = RknnModel::load_with_options(path, options)?;
        if model.input_info().len() != 2 || model.output_info().len() != 1 {
            return Err(Error::UnexpectedLayout(format!(
                "stereo graph must have two inputs and one output, got {} inputs and {} outputs",
                model.input_info().len(),
                model.output_info().len()
            )));
        }
        let input_shape = model.input_info()[0].shape.clone();
        let [1, height, width, 3] = input_shape.as_slice() else {
            return Err(Error::UnexpectedLayout(format!(
                "first input must be NHWC [1,H,W,3], got {input_shape:?}"
            )));
        };
        for input in model.input_info() {
            if input.shape != input_shape {
                return Err(Error::UnexpectedLayout(format!(
                    "left/right inputs differ: {:?} vs {:?}",
                    input_shape, input.shape
                )));
            }
        }
        Ok(Self {
            model,
            height: *height,
            width: *width,
            left_buffer: F16Image::zeroed(*height, *width)?,
            right_buffer: F16Image::zeroed(*height, *width)?,
        })
    }

    /// Return model input dimensions as `(width, height)`.
    pub fn input_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Run prepared FP16 NHWC inputs without allocating a per-frame conversion.
    pub fn infer(&mut self, left: &F16Image, right: &F16Image) -> Result<Tensor> {
        self.validate_image(left, "left")?;
        self.validate_image(right, "right")?;
        let mut outputs = self.model.run_nhwc_f16(&[&left.data, &right.data])?;
        reshape_disparity(outputs.remove(0), self.height, self.width)
    }

    /// Pack two RGB `[1,3,H,W]` tensors and run inference.
    pub fn infer_nchw(&mut self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        self.left_buffer.copy_from_nchw(left)?;
        self.right_buffer.copy_from_nchw(right)?;
        let mut outputs = self
            .model
            .run_nhwc_f16(&[&self.left_buffer.data, &self.right_buffer.data])?;
        reshape_disparity(outputs.remove(0), self.height, self.width)
    }

    /// Pack two HWC RGB8 images and run inference.
    pub fn infer_rgb8(&mut self, left: &[u8], right: &[u8]) -> Result<Tensor> {
        self.left_buffer.copy_from_rgb8(left)?;
        self.right_buffer.copy_from_rgb8(right)?;
        let mut outputs = self
            .model
            .run_nhwc_f16(&[&self.left_buffer.data, &self.right_buffer.data])?;
        reshape_disparity(outputs.remove(0), self.height, self.width)
    }

    fn validate_image(&self, image: &F16Image, label: &str) -> Result<()> {
        if image.height != self.height || image.width != self.width {
            return Err(Error::InvalidTensor(format!(
                "{label} image is {}x{}, model expects {}x{}",
                image.width, image.height, self.width, self.height
            )));
        }
        Ok(())
    }
}

fn reshape_disparity(value: Tensor, height: usize, width: usize) -> Result<Tensor> {
    if value.data.len() != height * width {
        return Err(Error::UnexpectedLayout(format!(
            "disparity has {} values, expected {}x{}",
            value.data.len(),
            width,
            height
        )));
    }
    Tensor::named(value.name, vec![1, 1, height, width], value.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nchw_is_packed_to_rgb_nhwc_f16() {
        let image = Tensor::new(vec![1, 3, 1, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let packed = F16Image::from_nchw(&image).unwrap();
        assert_eq!(packed.data, [0x3c00, 0x4000, 0x4200]);
    }

    #[test]
    fn rgb8_size_is_validated() {
        let image = F16Image::from_rgb8(1, 2, &[0, 127, 255, 1, 2, 3]).unwrap();
        assert_eq!(image.data.len(), 6);

        let error = F16Image::from_rgb8(1, 2, &[0, 1]).unwrap_err();
        assert!(matches!(error, Error::InvalidTensor(_)));
    }
}
