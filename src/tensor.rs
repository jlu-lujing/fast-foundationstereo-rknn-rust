use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tensor {
    pub name: Option<String>,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Self> {
        Self::named(None, shape, data)
    }

    pub fn named(
        name: Option<String>,
        shape: impl Into<Vec<usize>>,
        data: Vec<f32>,
    ) -> Result<Self> {
        let shape = shape.into();
        if shape.is_empty() || shape.contains(&0) {
            return Err(Error::InvalidTensor(format!(
                "shape must be non-empty and contain no zero dimensions: {shape:?}"
            )));
        }
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(Error::TensorElementCount {
                name: name.clone().unwrap_or_else(|| "<unnamed>".into()),
                shape,
                actual: data.len(),
                expected,
            });
        }
        Ok(Self { name, shape, data })
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn name_or<'a>(&'a self, fallback: &'a str) -> &'a str {
        self.name.as_deref().unwrap_or(fallback)
    }

    /// Convert this tensor to the byte order expected by rknnlite-style inputs.
    ///
    /// RKNN declares rank-4 tensors as NCHW but accepts NHWC-ordered bytes.
    /// Rank-5 tensors are passed through unchanged.
    pub fn to_rknn_input_f32(&self) -> Vec<f32> {
        if self.rank() != 4 {
            return self.data.clone();
        }
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let mut output = vec![0.0; self.data.len()];
        for batch in 0..n {
            for channel in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let src = ((batch * c + channel) * h + y) * w + x;
                        let dst = ((batch * h + y) * w + x) * c + channel;
                        output[dst] = self.data[src];
                    }
                }
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank4_uses_nhwc_byte_order() {
        let tensor = Tensor::new(vec![1, 2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(tensor.to_rknn_input_f32(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn rank5_passes_through() {
        let tensor = Tensor::new(vec![1, 1, 2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(tensor.to_rknn_input_f32(), tensor.data);
    }
}
