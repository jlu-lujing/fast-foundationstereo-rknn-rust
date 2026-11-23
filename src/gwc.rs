//! Group-wise correlation volume.
//!
//! `out[g,d,h,w] = sum_k L[gK+k,h,w] R[gK+k,h,w-d] / (||L|| ||R|| + eps)`.

use crate::{Error, Result, Tensor};

pub trait GwcBackend {
    fn volume(
        &self,
        left: &Tensor,
        right: &Tensor,
        groups: usize,
        max_disp: usize,
    ) -> Result<Tensor>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ScalarGwc;

impl GwcBackend for ScalarGwc {
    fn volume(
        &self,
        left: &Tensor,
        right: &Tensor,
        groups: usize,
        max_disp: usize,
    ) -> Result<Tensor> {
        groupwise_correlation_scalar(left, right, groups, max_disp)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AutoGwc;

impl GwcBackend for AutoGwc {
    fn volume(
        &self,
        left: &Tensor,
        right: &Tensor,
        groups: usize,
        max_disp: usize,
    ) -> Result<Tensor> {
        #[cfg(target_arch = "aarch64")]
        {
            groupwise_correlation_simd(left, right, groups, max_disp)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            groupwise_correlation_scalar(left, right, groups, max_disp)
        }
    }
}

pub fn groupwise_correlation_simd(
    left: &Tensor,
    right: &Tensor,
    groups: usize,
    max_disp: usize,
) -> Result<Tensor> {
    let (channels, height, width) = validate_inputs(left, right, groups, max_disp)?;
    let mut output = vec![0.0f32; groups * max_disp * height * width];
    unsafe extern "C" {
        fn gwc_volume_neon(
            left: *const f32,
            right: *const f32,
            output: *mut f32,
            channels: i32,
            height: i32,
            width: i32,
            groups: i32,
            max_disp: i32,
        ) -> i32;
    }
    let code = unsafe {
        gwc_volume_neon(
            left.data.as_ptr(),
            right.data.as_ptr(),
            output.as_mut_ptr(),
            channels as i32,
            height as i32,
            width as i32,
            groups as i32,
            max_disp as i32,
        )
    };
    if code != 0 {
        return Err(Error::InvalidTensor(format!(
            "SIMD GWC kernel returned {code}"
        )));
    }
    Tensor::new(vec![1, groups, max_disp, height, width], output)
}

pub fn groupwise_correlation_scalar(
    left: &Tensor,
    right: &Tensor,
    groups: usize,
    max_disp: usize,
) -> Result<Tensor> {
    let channels_height_width = |value: &Tensor| -> Result<(usize, usize, usize)> {
        let dims = match value.shape.as_slice() {
            [1, c, h, w] => (*c, *h, *w),
            [c, h, w] => (*c, *h, *w),
            _ => {
                return Err(Error::InvalidTensor(format!(
                    "GWC expects [1,C,H,W] or [C,H,W], got {:?}",
                    value.shape
                )));
            }
        };
        Ok(dims)
    };

    let (c, h, w) = channels_height_width(left)?;
    if channels_height_width(right)? != (c, h, w) {
        return Err(Error::InvalidTensor(format!(
            "left/right feature shapes differ: {:?} vs {:?}",
            left.shape, right.shape
        )));
    }
    if groups == 0 || c % groups != 0 || max_disp == 0 {
        return Err(Error::InvalidTensor(format!(
            "invalid GWC dimensions: C={c}, groups={groups}, max_disp={max_disp}"
        )));
    }

    let channels_per_group = c / groups;
    let hw = h * w;
    let mut output = vec![0.0f32; groups * max_disp * hw];
    for group in 0..groups {
        for disparity in 0..max_disp {
            for y in 0..h {
                for x in 0..w {
                    let shifted_x = x as isize - disparity as isize;
                    let mut dot = 0.0f32;
                    let mut left_norm = 0.0f32;
                    let mut right_norm = 0.0f32;
                    for inner in 0..channels_per_group {
                        let channel = group * channels_per_group + inner;
                        let left_index = (channel * h + y) * w + x;
                        let left_value = left.data[left_index];
                        let right_value = if shifted_x >= 0 {
                            right.data[(channel * h + y) * w + shifted_x as usize]
                        } else {
                            0.0
                        };
                        dot += left_value * right_value;
                        left_norm += left_value * left_value;
                        right_norm += right_value * right_value;
                    }
                    let denominator = left_norm.sqrt() * right_norm.sqrt() + 1e-5;
                    let index = ((group * max_disp + disparity) * h + y) * w + x;
                    output[index] = dot / denominator;
                }
            }
        }
    }
    Tensor::new(vec![1, groups, max_disp, h, w], output)
}

#[cfg(target_arch = "aarch64")]
pub fn groupwise_correlation_neon(
    left: &Tensor,
    right: &Tensor,
    groups: usize,
    max_disp: usize,
) -> Result<Tensor> {
    use std::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vsqrtq_f32, vst1q_f32};

    let (channels, height, width) = validate_inputs(left, right, groups, max_disp)?;
    let channels_per_group = channels / groups;
    let hw = height * width;
    let mut output = vec![0.0f32; groups * max_disp * hw];

    unsafe {
        for group in 0..groups {
            let group_channel = group * channels_per_group;
            for y in 0..height {
                let mut left_norm = vec![0.0f32; width];
                for inner in 0..channels_per_group {
                    let channel = group_channel + inner;
                    let row = left.data.as_ptr().add((channel * height + y) * width);
                    let mut x = 0;
                    while x + 4 <= width {
                        let value = vld1q_f32(row.add(x));
                        let current = vld1q_f32(left_norm.as_ptr().add(x));
                        vst1q_f32(
                            left_norm.as_mut_ptr().add(x),
                            vfmaq_f32(current, value, value),
                        );
                        x += 4;
                    }
                    for (offset, value) in left_norm[x..].iter_mut().enumerate() {
                        let column = x + offset;
                        *value += left.data[(channel * height + y) * width + column].powi(2);
                    }
                }
                let mut x = 0;
                while x + 4 <= width {
                    let norm = vld1q_f32(left_norm.as_ptr().add(x));
                    vst1q_f32(left_norm.as_mut_ptr().add(x), vsqrtq_f32(norm));
                    x += 4;
                }
                for value in left_norm.iter_mut().skip(x) {
                    *value = value.sqrt();
                }

                for disparity in 0..max_disp {
                    let mut dot = vec![0.0f32; width];
                    let mut right_norm = vec![0.0f32; width];
                    for inner in 0..channels_per_group {
                        let channel = group_channel + inner;
                        let left_row = left.data.as_ptr().add((channel * height + y) * width);
                        let right_row = right.data.as_ptr().add((channel * height + y) * width);
                        let mut x = disparity;
                        while x + 4 <= width {
                            let l = vld1q_f32(left_row.add(x));
                            let r = vld1q_f32(right_row.add(x - disparity));
                            vst1q_f32(
                                dot.as_mut_ptr().add(x),
                                vfmaq_f32(vld1q_f32(dot.as_ptr().add(x)), l, r),
                            );
                            vst1q_f32(
                                right_norm.as_mut_ptr().add(x),
                                vfmaq_f32(vld1q_f32(right_norm.as_ptr().add(x)), r, r),
                            );
                            x += 4;
                        }
                        for x in x..width {
                            let lv = left.data[(channel * height + y) * width + x];
                            let rv = right.data[(channel * height + y) * width + x - disparity];
                            dot[x] += lv * rv;
                            right_norm[x] += rv * rv;
                        }
                    }

                    let destination = (group * max_disp + disparity) * hw + y * width;
                    let epsilon = vdupq_n_f32(1e-5);
                    let mut x = disparity;
                    while x + 4 <= width {
                        let denominator = vfmaq_f32(
                            epsilon,
                            vld1q_f32(left_norm.as_ptr().add(x)),
                            vsqrtq_f32(vld1q_f32(right_norm.as_ptr().add(x))),
                        );
                        vst1q_f32(output.as_mut_ptr().add(destination + x), {
                            use std::arch::aarch64::vdivq_f32;
                            vdivq_f32(vld1q_f32(dot.as_ptr().add(x)), denominator)
                        });
                        x += 4;
                    }
                    for x in x..width {
                        output[destination + x] =
                            dot[x] / (left_norm[x] * right_norm[x].sqrt() + 1e-5);
                    }
                }
            }
        }
    }
    Tensor::new(vec![1, groups, max_disp, height, width], output)
}

fn validate_inputs(
    left: &Tensor,
    right: &Tensor,
    groups: usize,
    max_disp: usize,
) -> Result<(usize, usize, usize)> {
    let dims = |value: &Tensor| -> Result<(usize, usize, usize)> {
        match value.shape.as_slice() {
            [1, c, h, w] => Ok((*c, *h, *w)),
            [c, h, w] => Ok((*c, *h, *w)),
            _ => Err(Error::InvalidTensor(format!(
                "GWC expects [1,C,H,W] or [C,H,W], got {:?}",
                value.shape
            ))),
        }
    };
    let left_dims = dims(left)?;
    if dims(right)? != left_dims {
        return Err(Error::InvalidTensor(format!(
            "left/right feature shapes differ: {:?} vs {:?}",
            left.shape, right.shape
        )));
    }
    let (channels, _, _) = left_dims;
    if groups == 0 || channels % groups != 0 || max_disp == 0 {
        return Err(Error::InvalidTensor(format!(
            "invalid GWC dimensions: C={channels}, groups={groups}, max_disp={max_disp}"
        )));
    }
    Ok(left_dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gwc_reference_small_case() {
        let left = Tensor::new(
            vec![1, 4, 1, 3],
            vec![1.0, 2.0, 3.0, 0.5, 1.0, 1.5, 2.0, 1.0, 0.0, 0.3, 0.2, 0.1],
        )
        .unwrap();
        let right = left.clone();
        let output = groupwise_correlation_scalar(&left, &right, 2, 2).unwrap();
        assert_eq!(output.shape, vec![1, 2, 2, 1, 3]);
        assert!((output.data[0] - 1.0).abs() < 1e-5);
        assert!((output.data[3] - 0.0).abs() < 1e-6);
        assert!((output.data[4] - 1.0).abs() < 1e-5);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar() {
        let left = Tensor::new(
            vec![1, 8, 3, 7],
            (0..8 * 3 * 7)
                .map(|index| ((index * 37 % 23) as f32 - 11.0) / 13.0)
                .collect(),
        )
        .unwrap();
        let right = Tensor::new(
            vec![1, 8, 3, 7],
            (0..8 * 3 * 7)
                .map(|index| ((index * 19 % 29) as f32 - 14.0) / 17.0)
                .collect(),
        )
        .unwrap();
        let scalar = groupwise_correlation_scalar(&left, &right, 2, 5).unwrap();
        let neon = groupwise_correlation_neon(&left, &right, 2, 5).unwrap();
        for (actual, expected) in neon.data.iter().zip(&scalar.data) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn simd_kernel_matches_scalar() {
        let left = Tensor::new(
            vec![1, 8, 3, 7],
            (0..8 * 3 * 7)
                .map(|index| ((index * 13 % 31) as f32 - 15.0) / 11.0)
                .collect(),
        )
        .unwrap();
        let right = Tensor::new(
            vec![1, 8, 3, 7],
            (0..8 * 3 * 7)
                .map(|index| ((index * 17 % 37) as f32 - 18.0) / 19.0)
                .collect(),
        )
        .unwrap();
        let scalar = groupwise_correlation_scalar(&left, &right, 2, 5).unwrap();
        let simd = groupwise_correlation_simd(&left, &right, 2, 5).unwrap();
        for (actual, expected) in simd.data.iter().zip(&scalar.data) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}
