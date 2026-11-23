use std::path::{Path, PathBuf};

use crate::{Error, Result, Tensor};

pub const RESIZE_INPUT: &str = "/cost_agg/post8_to_4/upsample/upsample.1/Transpose_1_output_0";
pub const INITIAL_DISPARITY: &str = "/ReduceSum_output_0";
pub const P8: &str = "/cost_agg/post8_to_4/out.1/relu_1/Relu_output_0";

#[derive(Debug, Clone)]
pub struct OrtOptions {
    pub dylib_path: Option<PathBuf>,
    pub refine_operator_library: Option<PathBuf>,
    pub resize_threads: usize,
    pub refine_threads: usize,
}

impl Default for OrtOptions {
    fn default() -> Self {
        Self {
            dylib_path: None,
            refine_operator_library: None,
            resize_threads: 1,
            refine_threads: 4,
        }
    }
}

pub struct OrtCpuStages {
    _environment: ort::environment::Environment,
    resize: ort::session::Session,
    refine: ort::session::Session,
}

impl OrtCpuStages {
    pub fn load(
        resize_model: impl AsRef<Path>,
        refine_model: impl AsRef<Path>,
        options: &OrtOptions,
    ) -> Result<Self> {
        let environment = if let Some(path) = options.dylib_path.as_deref() {
            ort::init_from(path)
                .map_err(|error| Error::Ort(error.to_string()))?
                .build()
                .map_err(|error| Error::Ort(error.to_string()))?
        } else {
            ort::init()
                .build()
                .map_err(|error| Error::Ort(error.to_string()))?
        };
        let resize = ort::session::Session::builder(&environment)
            .map_err(ort_string)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level2)
            .map_err(ort_string)?
            .with_intra_threads(options.resize_threads)
            .map_err(ort_string)?
            .commit_from_file(resize_model.as_ref())
            .map_err(ort_string)?;
        let mut refine_builder = ort::session::Session::builder(&environment)
            .map_err(ort_string)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level2)
            .map_err(ort_string)?;
        if let Some(path) = options.refine_operator_library.as_deref() {
            refine_builder = refine_builder
                .with_operator_library(path)
                .map_err(ort_string)?;
        }
        let refine = refine_builder
            .with_intra_threads(options.refine_threads)
            .map_err(ort_string)?
            .commit_from_file(refine_model.as_ref())
            .map_err(ort_string)?;
        Ok(Self {
            _environment: environment,
            resize,
            refine,
        })
    }

    pub fn resize(&mut self, attention: &Tensor) -> Result<Tensor> {
        let input = ort_tensor(attention)?;
        let outputs = self
            .resize
            .run(ort::inputs![RESIZE_INPUT => input])
            .map_err(ort_string)?;
        extract_first(&outputs, "stage_resize_only")
    }

    pub fn refine(
        &mut self,
        initial_disparity: &Tensor,
        p8: &Tensor,
        features_left_04: &Tensor,
        features_right_04: &Tensor,
        stem_2x: &Tensor,
    ) -> Result<Tensor> {
        let outputs = self
            .refine
            .run(ort::inputs![
                INITIAL_DISPARITY => ort_tensor(initial_disparity)?,
                P8 => ort_tensor(p8)?,
                "features_left_04" => ort_tensor(features_left_04)?,
                "features_right_04" => ort_tensor(features_right_04)?,
                "stem_2x" => ort_tensor(stem_2x)?,
            ])
            .map_err(ort_string)?;
        extract_first(&outputs, "segment_refine")
    }
}

fn ort_tensor(value: &Tensor) -> Result<ort::value::Tensor<f32>> {
    ort::value::Tensor::from_array((value.shape.clone(), value.data.clone().into_boxed_slice()))
        .map_err(ort_string)
}

fn extract_first(outputs: &ort::session::SessionOutputs<'_>, graph: &str) -> Result<Tensor> {
    if outputs.len() == 0 {
        return Err(Error::UnexpectedLayout(format!(
            "{graph} returned no outputs"
        )));
    }
    let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_string)?;
    Tensor::new(
        shape
            .iter()
            .map(|dimension| *dimension as usize)
            .collect::<Vec<_>>(),
        data.to_vec(),
    )
}

fn ort_string(error: impl std::fmt::Display) -> Error {
    Error::Ort(error.to_string())
}
