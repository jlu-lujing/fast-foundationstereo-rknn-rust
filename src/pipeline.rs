use std::path::{Path, PathBuf};

use crate::{
    Error, Result, Tensor,
    cpu::{OrtCpuStages, OrtOptions},
    gwc::{AutoGwc, GwcBackend},
    rknn::{NpuCore, RknnModel, RknnOptions},
};

pub const PRE_INPUT_NAMES: [&str; 6] = [
    "features_left_04",
    "features_left_08",
    "features_left_16",
    "features_left_32",
    "features_right_04",
    "gwc_volume",
];
pub const FEATURE_OUTPUT_NAMES: [&str; 5] = [
    "features_left_04",
    "features_left_08",
    "features_left_16",
    "features_left_32",
    "stem_2x",
];

#[derive(Debug, Clone)]
pub struct ModelArtifacts {
    pub root: PathBuf,
    pub feature: PathBuf,
    pub pre_resize: PathBuf,
    pub post_resize: PathBuf,
    pub resize_onnx: PathBuf,
    pub refine_onnx: PathBuf,
}

impl ModelArtifacts {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            feature: root.join("feature_single.rknn"),
            pre_resize: root.join("segment_pre_resize.rknn"),
            post_resize: root.join("segment_post_resize_full.rknn"),
            resize_onnx: root.join("stage_resize_only.onnx"),
            refine_onnx: root.join("segment_refine.onnx"),
            root,
        }
    }

    pub fn prefer_ffs_refine(&mut self) -> bool {
        for name in ["segment_refine_native_ffs.onnx", "segment_refine_ffs.onnx"] {
            let path = self.root.join(name);
            if path.is_file() {
                self.refine_onnx = path;
                return true;
            }
        }
        false
    }

    pub fn validate(&self) -> Result<()> {
        for path in [
            &self.feature,
            &self.pre_resize,
            &self.post_resize,
            &self.resize_onnx,
            &self.refine_onnx,
        ] {
            if !path.is_file() {
                return Err(Error::io(
                    path,
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "required model artifact does not exist",
                    ),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PipelineOptions {
    pub artifacts: ModelArtifacts,
    pub rknn_library: Option<PathBuf>,
    pub feature_core: NpuCore,
    pub pre_core: NpuCore,
    pub post_core: NpuCore,
    pub ort: OrtOptions,
}

impl PipelineOptions {
    pub fn new(artifacts: impl Into<PathBuf>) -> Self {
        Self {
            artifacts: ModelArtifacts::new(artifacts),
            rknn_library: None,
            feature_core: NpuCore::Core1,
            pre_core: NpuCore::Core01,
            post_core: NpuCore::Core2,
            ort: OrtOptions::default(),
        }
    }
}

pub struct FoundationStereo {
    feature: RknnModel,
    pre_resize: RknnModel,
    post_resize: RknnModel,
    cpu: OrtCpuStages,
    gwc: AutoGwc,
    max_disp: usize,
    groups: usize,
}

#[derive(Debug, Clone)]
pub struct PipelineTrace {
    pub left_features: Vec<Tensor>,
    pub right_features: Vec<Tensor>,
    pub volume: Tensor,
    pub pre_outputs: Vec<Tensor>,
    pub resized: Tensor,
    pub post_outputs: Vec<Tensor>,
    pub disparity: Tensor,
}

impl FoundationStereo {
    pub fn load(options: PipelineOptions) -> Result<Self> {
        options.artifacts.validate()?;
        let rknn = |path: &Path, core| {
            RknnModel::load_with_options(
                path,
                RknnOptions {
                    library_path: options.rknn_library.clone(),
                    core,
                },
            )
        };
        let feature = rknn(&options.artifacts.feature, options.feature_core)?;
        let pre_resize = rknn(&options.artifacts.pre_resize, options.pre_core)?;
        let post_resize = rknn(&options.artifacts.post_resize, options.post_core)?;
        let cpu = OrtCpuStages::load(
            &options.artifacts.resize_onnx,
            &options.artifacts.refine_onnx,
            &options.ort,
        )?;
        let volume_shape = pre_resize
            .input_info()
            .iter()
            .find(|info| info.name == "gwc_volume")
            .map(|info| info.shape.clone())
            .ok_or_else(|| Error::MissingOutput {
                name: "gwc_volume model input".to_string(),
            })?;
        let (groups, max_disp) = match volume_shape.as_slice() {
            [1, groups, max_disp, _, _] => (*groups, *max_disp),
            _ => {
                return Err(Error::UnexpectedLayout(format!(
                    "gwc_volume input must be [1,G,D,H,W], got {volume_shape:?}"
                )));
            }
        };
        Ok(Self {
            feature,
            pre_resize,
            post_resize,
            cpu,
            gwc: AutoGwc,
            max_disp,
            groups,
        })
    }

    pub fn with_cost_volume(mut self, groups: usize, max_disp: usize) -> Result<Self> {
        if groups == 0 || groups > 32 || max_disp == 0 {
            return Err(Error::InvalidTensor(format!(
                "invalid cost-volume dimensions: groups={groups}, max_disp={max_disp}"
            )));
        }
        self.groups = groups;
        self.max_disp = max_disp;
        Ok(self)
    }

    pub fn infer(&mut self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        Ok(self.infer_trace(left, right)?.disparity)
    }

    pub fn infer_trace(&mut self, left: &Tensor, right: &Tensor) -> Result<PipelineTrace> {
        validate_image(left, "left")?;
        validate_image(right, "right")?;
        if left.shape != right.shape {
            return Err(Error::InvalidTensor(format!(
                "left/right image shapes differ: {:?} vs {:?}",
                left.shape, right.shape
            )));
        }

        let left_features = self.feature.run(std::slice::from_ref(left))?;
        let right_features = self.feature.run(std::slice::from_ref(right))?;
        let left_04 = feature_output(&left_features, "features_left_04", 0)?;
        let left_08 = feature_output(&left_features, "features_left_08", 1)?;
        let left_16 = feature_output(&left_features, "features_left_16", 2)?;
        let left_32 = feature_output(&left_features, "features_left_32", 3)?;
        let stem_2x = feature_output(&left_features, "stem_2x", 4)?;
        let right_04 = feature_output(&right_features, "features_left_04", 0)?;

        let volume = self
            .gwc
            .volume(left_04, right_04, self.groups, self.max_disp)?;
        let pre_inputs = [
            left_04.clone(),
            left_08.clone(),
            left_16.clone(),
            left_32.clone(),
            right_04.clone(),
            volume.clone(),
        ];
        let pre_outputs = self.pre_resize.run(&pre_inputs)?;
        if pre_outputs.len() != 2 {
            return Err(Error::UnexpectedLayout(format!(
                "pre-resize model returned {} tensors, expected 2",
                pre_outputs.len()
            )));
        }
        let attention = &pre_outputs[0];
        let corr_attention = &pre_outputs[1];
        let resized = self.cpu.resize(attention)?;
        let post_outputs = self
            .post_resize
            .run(&[resized.clone(), corr_attention.clone()])?;
        if post_outputs.len() != 2 {
            return Err(Error::UnexpectedLayout(format!(
                "post-resize model returned {} tensors, expected 2",
                post_outputs.len()
            )));
        }
        let p8 = &post_outputs[0];
        let initial_disparity = &post_outputs[1];
        let disparity = self
            .cpu
            .refine(initial_disparity, p8, left_04, right_04, stem_2x)?;
        Ok(PipelineTrace {
            left_features,
            right_features,
            volume,
            pre_outputs,
            resized,
            post_outputs,
            disparity,
        })
    }
}

fn validate_image(value: &Tensor, label: &str) -> Result<()> {
    if value.rank() != 4 || value.shape[0] != 1 || value.shape[1] != 3 {
        return Err(Error::InvalidTensor(format!(
            "{label} image must be NCHW [1,3,H,W], got {:?}",
            value.shape
        )));
    }
    Ok(())
}

fn feature_output<'a>(
    outputs: &'a [Tensor],
    name: &str,
    fallback_index: usize,
) -> Result<&'a Tensor> {
    outputs
        .iter()
        .find(|tensor| tensor.name.as_deref() == Some(name))
        .or_else(|| outputs.get(fallback_index))
        .ok_or_else(|| Error::MissingOutput {
            name: name.to_string(),
        })
}
