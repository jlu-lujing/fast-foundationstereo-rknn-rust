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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_element_count() {
        let error = Tensor::new(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(error, Error::TensorElementCount { .. }));
    }

    #[test]
    fn exposes_shape_metadata() {
        let tensor = Tensor::new(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.rank(), 4);
        assert_eq!(tensor.name_or("disparity"), "disparity");
    }
}
