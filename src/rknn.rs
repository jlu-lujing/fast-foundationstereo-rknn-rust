use std::{
    ffi::c_void,
    path::{Path, PathBuf},
    ptr,
    sync::Arc,
};

use libloading::Library;

use crate::{Error, Result, Tensor};

#[cfg(target_pointer_width = "32")]
type RknnContext = u32;
#[cfg(not(target_pointer_width = "32"))]
type RknnContext = u64;

const RKNN_QUERY_IN_OUT_NUM: u32 = 0;
const RKNN_QUERY_INPUT_ATTR: u32 = 1;
const RKNN_QUERY_OUTPUT_ATTR: u32 = 2;
const RKNN_TENSOR_FLOAT16: u32 = 1;
const RKNN_TENSOR_NHWC: u32 = 1;
const RKNN_MAX_DIMS: usize = 16;
const RKNN_MAX_NAME_LEN: usize = 256;

#[repr(C)]
struct RknnInputOutputNum {
    n_input: u32,
    n_output: u32,
}

#[repr(C)]
#[derive(Clone)]
struct RknnTensorAttr {
    index: u32,
    n_dims: u32,
    dims: [u32; RKNN_MAX_DIMS],
    name: [u8; RKNN_MAX_NAME_LEN],
    n_elems: u32,
    size: u32,
    format: u32,
    data_type: u32,
    quant_type: u32,
    fl: i8,
    zp: i32,
    scale: f32,
    w_stride: u32,
    size_with_stride: u32,
    pass_through: u8,
    h_stride: u32,
}

impl RknnTensorAttr {
    fn new() -> Self {
        unsafe { std::mem::zeroed() }
    }

    fn name_str(&self) -> &str {
        std::str::from_utf8(&self.name)
            .unwrap_or("")
            .trim_end_matches('\0')
    }
}

#[repr(C)]
struct RknnInput {
    index: u32,
    buf: *mut c_void,
    size: u32,
    pass_through: u8,
    type_: u32,
    format: u32,
}

#[repr(C)]
struct RknnOutput {
    want_float: u8,
    is_prealloc: u8,
    index: u32,
    buf: *mut c_void,
    size: u32,
}

type FnInit = unsafe extern "C" fn(*mut RknnContext, *const c_void, u32, u32, *const c_void) -> i32;
type FnQuery = unsafe extern "C" fn(RknnContext, u32, *mut c_void, u32) -> i32;
type FnRun = unsafe extern "C" fn(RknnContext, *const c_void) -> i32;
type FnDestroy = unsafe extern "C" fn(RknnContext) -> i32;
type FnInputsSet = unsafe extern "C" fn(RknnContext, u32, *mut RknnInput) -> i32;
type FnOutputsGet = unsafe extern "C" fn(RknnContext, u32, *mut RknnOutput, *const c_void) -> i32;
type FnOutputsRelease = unsafe extern "C" fn(RknnContext, u32, *mut RknnOutput) -> i32;
type FnSetCoreMask = unsafe extern "C" fn(RknnContext, u32) -> i32;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum NpuCore {
    #[default]
    Auto,
    Core0,
    Core1,
    Core2,
    Core01,
    All,
}

impl NpuCore {
    const fn mask(self) -> u32 {
        match self {
            Self::Auto => 0,
            Self::Core0 => 1,
            Self::Core1 => 2,
            Self::Core2 => 4,
            Self::Core01 => 3,
            Self::All => 0x7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub index: u32,
    pub name: String,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

#[derive(Debug, Clone)]
pub struct RknnOptions {
    pub library_path: Option<PathBuf>,
    pub core: NpuCore,
}

impl Default for RknnOptions {
    fn default() -> Self {
        Self {
            library_path: None,
            core: NpuCore::All,
        }
    }
}

struct RknnFunctions {
    init: FnInit,
    query: FnQuery,
    run: FnRun,
    destroy: FnDestroy,
    inputs_set: FnInputsSet,
    outputs_get: FnOutputsGet,
    outputs_release: FnOutputsRelease,
    set_core_mask: FnSetCoreMask,
}

struct RknnLibrary {
    _library: Library,
    functions: RknnFunctions,
}

impl RknnLibrary {
    fn open(path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(path) }.map_err(|error| Error::RknnSymbol {
            symbol: path.display().to_string(),
            message: error.to_string(),
        })?;
        unsafe {
            Ok(Self {
                functions: RknnFunctions {
                    init: *load_symbol(&library, b"rknn_init\0")?,
                    query: *load_symbol(&library, b"rknn_query\0")?,
                    run: *load_symbol(&library, b"rknn_run\0")?,
                    destroy: *load_symbol(&library, b"rknn_destroy\0")?,
                    inputs_set: *load_symbol(&library, b"rknn_inputs_set\0")?,
                    outputs_get: *load_symbol(&library, b"rknn_outputs_get\0")?,
                    outputs_release: *load_symbol(&library, b"rknn_outputs_release\0")?,
                    set_core_mask: *load_symbol(&library, b"rknn_set_core_mask\0")?,
                },
                _library: library,
            })
        }
    }
}

unsafe fn load_symbol<'library, T>(
    library: &'library Library,
    name: &[u8],
) -> Result<libloading::Symbol<'library, T>> {
    unsafe { library.get::<T>(name) }.map_err(|error| Error::RknnSymbol {
        symbol: String::from_utf8_lossy(name)
            .trim_end_matches('\0')
            .to_string(),
        message: error.to_string(),
    })
}

fn resolve_library(explicit: Option<&Path>) -> Result<PathBuf> {
    let mut tried = Vec::new();
    if let Some(path) = explicit {
        tried.push(path.to_path_buf());
        if path.exists() {
            return Ok(path.to_path_buf());
        }
    }
    if let Some(path) = std::env::var_os("RKNN_LIBRARY").map(PathBuf::from) {
        tried.push(path.clone());
        if path.exists() {
            return Ok(path);
        }
    }
    for candidate in [
        PathBuf::from("/usr/lib/librknnrt.so"),
        PathBuf::from("/usr/local/lib/librknnrt.so"),
    ] {
        tried.push(candidate.clone());
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    // The dynamic loader may still resolve this name even when path lookup fails.
    tried.push(PathBuf::from("librknnrt.so"));
    Ok(PathBuf::from("librknnrt.so"))
}

pub struct RknnModel {
    context: RknnContext,
    library: Arc<RknnLibrary>,
    _model_data: Vec<u8>,
    input_info: Vec<TensorInfo>,
    input_attrs: Vec<RknnTensorAttr>,
    output_info: Vec<TensorInfo>,
    output_attrs: Vec<RknnTensorAttr>,
}

unsafe impl Send for RknnModel {}

impl RknnModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_options(path, RknnOptions::default())
    }

    pub fn load_with_options(path: impl AsRef<Path>, options: RknnOptions) -> Result<Self> {
        let model_path = path.as_ref();
        let model_data = std::fs::read(model_path).map_err(|error| Error::io(model_path, error))?;
        validate_model_bytes(model_path, &model_data)?;
        let library_path = resolve_library(options.library_path.as_deref())?;
        let library = Arc::new(RknnLibrary::open(&library_path)?);
        let mut context: RknnContext = 0;
        let code = unsafe {
            (library.functions.init)(
                &mut context,
                model_data.as_ptr().cast(),
                model_data.len() as u32,
                0,
                ptr::null(),
            )
        };
        if code != 0 {
            return Err(Error::RknnCall {
                operation: "init",
                code,
            });
        }
        let code = unsafe { (library.functions.set_core_mask)(context, options.core.mask()) };
        if code != 0 {
            unsafe {
                (library.functions.destroy)(context);
            }
            return Err(Error::RknnCall {
                operation: "set_core_mask",
                code,
            });
        }

        let io = match query_model_io(&library.functions, context) {
            Ok(value) => value,
            Err(error) => {
                unsafe {
                    (library.functions.destroy)(context);
                }
                return Err(error);
            }
        };
        Ok(Self {
            context,
            library,
            _model_data: model_data,
            input_info: io.input_info,
            input_attrs: io.input_attrs,
            output_info: io.output_info,
            output_attrs: io.output_attrs,
        })
    }

    pub fn input_info(&self) -> &[TensorInfo] {
        &self.input_info
    }

    pub fn output_info(&self) -> &[TensorInfo] {
        &self.output_info
    }

    /// Run a model whose inputs are already contiguous FP16 NHWC tensors.
    ///
    /// Pass-through mode avoids RKNN doing a second FP32-to-FP16 conversion for
    /// every frame.
    pub fn run_nhwc_f16(&mut self, inputs: &[&[u16]]) -> Result<Vec<Tensor>> {
        let mut raw_inputs = self.prepare_nhwc_f16_inputs(inputs)?;
        self.run_inputs(&mut raw_inputs)?;
        self.collect_outputs()
    }

    /// Run FP16 NHWC inputs into a caller-owned single FP32 output buffer.
    ///
    /// This uses RKNN's preallocated-output mode and avoids the runtime output
    /// allocation and a second copy into a `Tensor` on every frame.
    pub fn run_nhwc_f16_into(&mut self, inputs: &[&[u16]], output: &mut [f32]) -> Result<()> {
        if self.output_attrs.len() != 1 {
            return Err(Error::UnexpectedLayout(format!(
                "preallocated output requires one model output, got {}",
                self.output_attrs.len()
            )));
        }
        let attr = &self.output_attrs[0];
        let expected = attr.n_elems as usize;
        if output.len() != expected {
            return Err(Error::InvalidTensor(format!(
                "output has {} values, expected {expected}",
                output.len()
            )));
        }
        let mut raw_inputs = self.prepare_nhwc_f16_inputs(inputs)?;
        self.run_inputs(&mut raw_inputs)?;

        let mut raw_outputs = [RknnOutput {
            want_float: 1,
            is_prealloc: 1,
            index: 0,
            buf: output.as_mut_ptr().cast(),
            size: std::mem::size_of_val(output) as u32,
        }];
        let code = unsafe {
            (self.library.functions.outputs_get)(
                self.context,
                raw_outputs.len() as u32,
                raw_outputs.as_mut_ptr(),
                ptr::null(),
            )
        };
        check_call(code, "outputs_get")?;

        let output_result =
            if raw_outputs[0].buf.is_null() || raw_outputs[0].size != expected as u32 * 4 {
                Err(Error::UnexpectedLayout(format!(
                    "preallocated output returned buf={:p}, size={}",
                    raw_outputs[0].buf, raw_outputs[0].size
                )))
            } else {
                Ok(())
            };
        let code = unsafe {
            (self.library.functions.outputs_release)(
                self.context,
                raw_outputs.len() as u32,
                raw_outputs.as_mut_ptr(),
            )
        };
        output_result?;
        check_call(code, "outputs_release")
    }

    fn prepare_nhwc_f16_inputs(&self, inputs: &[&[u16]]) -> Result<Vec<RknnInput>> {
        if inputs.len() != self.input_attrs.len() {
            return Err(Error::InputCount {
                expected: self.input_attrs.len(),
                actual: inputs.len(),
            });
        }
        let mut raw_inputs = Vec::with_capacity(inputs.len());
        for (index, (buffer, attr)) in inputs.iter().zip(&self.input_attrs).enumerate() {
            if attr.data_type != RKNN_TENSOR_FLOAT16 || attr.format != RKNN_TENSOR_NHWC {
                return Err(Error::UnexpectedLayout(format!(
                    "input {index} must be FP16 NHWC, got type={} format={}",
                    attr.data_type, attr.format
                )));
            }
            if buffer.len() != attr.n_elems as usize {
                return Err(Error::InvalidTensor(format!(
                    "input {index} has {} FP16 values, expected {}",
                    buffer.len(),
                    attr.n_elems
                )));
            }
            raw_inputs.push(RknnInput {
                index: index as u32,
                buf: buffer.as_ptr().cast_mut().cast(),
                size: std::mem::size_of_val(*buffer) as u32,
                pass_through: 1,
                type_: RKNN_TENSOR_FLOAT16,
                format: RKNN_TENSOR_NHWC,
            });
        }
        Ok(raw_inputs)
    }

    fn run_inputs(&mut self, raw_inputs: &mut [RknnInput]) -> Result<()> {
        let code = unsafe {
            (self.library.functions.inputs_set)(
                self.context,
                raw_inputs.len() as u32,
                raw_inputs.as_mut_ptr(),
            )
        };
        check_call(code, "inputs_set")?;
        let code = unsafe { (self.library.functions.run)(self.context, ptr::null()) };
        check_call(code, "run")
    }

    fn collect_outputs(&mut self) -> Result<Vec<Tensor>> {
        let mut raw_outputs = self
            .output_attrs
            .iter()
            .enumerate()
            .map(|(index, _)| RknnOutput {
                want_float: 1,
                is_prealloc: 0,
                index: index as u32,
                buf: ptr::null_mut(),
                size: 0,
            })
            .collect::<Vec<_>>();
        let code = unsafe {
            (self.library.functions.outputs_get)(
                self.context,
                raw_outputs.len() as u32,
                raw_outputs.as_mut_ptr(),
                ptr::null(),
            )
        };
        check_call(code, "outputs_get")?;

        let output_result = (|| {
            let mut outputs = Vec::with_capacity(raw_outputs.len());
            for (index, output) in raw_outputs.iter().enumerate() {
                let attr = &self.output_attrs[index];
                let shape = attr.dims[..attr.n_dims as usize]
                    .iter()
                    .map(|value| *value as usize)
                    .collect::<Vec<_>>();
                let count = shape.iter().product::<usize>();
                let data = if output.buf.is_null() || count == 0 {
                    return Err(Error::UnexpectedLayout(format!(
                        "output {index} returned a null or empty buffer"
                    )));
                } else {
                    unsafe { std::slice::from_raw_parts(output.buf.cast::<f32>(), count) }.to_vec()
                };
                outputs.push(Tensor::named(
                    Some(attr.name_str().to_string()),
                    shape,
                    data,
                )?);
            }
            Ok(outputs)
        })();
        let code = unsafe {
            (self.library.functions.outputs_release)(
                self.context,
                raw_outputs.len() as u32,
                raw_outputs.as_mut_ptr(),
            )
        };
        let outputs = output_result?;
        check_call(code, "outputs_release")?;
        Ok(outputs)
    }
}

impl Drop for RknnModel {
    fn drop(&mut self) {
        unsafe {
            (self.library.functions.destroy)(self.context);
        }
    }
}

fn query_io_num(functions: &RknnFunctions, context: RknnContext) -> Result<RknnInputOutputNum> {
    let mut value = RknnInputOutputNum {
        n_input: 0,
        n_output: 0,
    };
    let code = unsafe {
        (functions.query)(
            context,
            RKNN_QUERY_IN_OUT_NUM,
            (&mut value as *mut RknnInputOutputNum).cast(),
            std::mem::size_of::<RknnInputOutputNum>() as u32,
        )
    };
    check_call(code, "query_in_out_num")?;
    Ok(value)
}

struct ModelIo {
    input_info: Vec<TensorInfo>,
    input_attrs: Vec<RknnTensorAttr>,
    output_info: Vec<TensorInfo>,
    output_attrs: Vec<RknnTensorAttr>,
}

fn query_model_io(functions: &RknnFunctions, context: RknnContext) -> Result<ModelIo> {
    let io_num = query_io_num(functions, context)?;
    let mut input_info = Vec::with_capacity(io_num.n_input as usize);
    let mut input_attrs = Vec::with_capacity(io_num.n_input as usize);
    for index in 0..io_num.n_input {
        let attr = query_attr(functions, context, RKNN_QUERY_INPUT_ATTR, index)?;
        input_info.push(info_from_attr(&attr));
        input_attrs.push(attr);
    }
    let mut output_info = Vec::with_capacity(io_num.n_output as usize);
    let mut output_attrs = Vec::with_capacity(io_num.n_output as usize);
    for index in 0..io_num.n_output {
        let attr = query_attr(functions, context, RKNN_QUERY_OUTPUT_ATTR, index)?;
        output_info.push(info_from_attr(&attr));
        output_attrs.push(attr);
    }
    Ok(ModelIo {
        input_info,
        input_attrs,
        output_info,
        output_attrs,
    })
}

fn query_attr(
    functions: &RknnFunctions,
    context: RknnContext,
    command: u32,
    index: u32,
) -> Result<RknnTensorAttr> {
    let mut attr = RknnTensorAttr::new();
    attr.index = index;
    let code = unsafe {
        (functions.query)(
            context,
            command,
            (&mut attr as *mut RknnTensorAttr).cast(),
            std::mem::size_of::<RknnTensorAttr>() as u32,
        )
    };
    check_call(
        code,
        if command == RKNN_QUERY_INPUT_ATTR {
            "query_input_attr"
        } else {
            "query_output_attr"
        },
    )?;
    Ok(attr)
}

fn info_from_attr(attr: &RknnTensorAttr) -> TensorInfo {
    TensorInfo {
        index: attr.index,
        name: attr.name_str().to_string(),
        shape: attr.dims[..attr.n_dims as usize]
            .iter()
            .map(|value| *value as usize)
            .collect(),
        bytes: attr.size as usize,
    }
}

fn check_call(code: i32, operation: &'static str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::RknnCall { operation, code })
    }
}

fn validate_model_bytes(path: &Path, data: &[u8]) -> Result<()> {
    if data.len() < 16 {
        return Err(Error::InvalidModel {
            path: path.to_path_buf(),
            message: format!("file is only {} bytes", data.len()),
        });
    }
    if &data[..4] != b"RKNN" {
        return Err(Error::InvalidModel {
            path: path.to_path_buf(),
            message: "missing RKNN magic header".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_and_non_rknn_models_before_runtime_init() {
        let path = Path::new("model.rknn");
        assert!(matches!(
            validate_model_bytes(path, &[]),
            Err(Error::InvalidModel { .. })
        ));
        assert!(matches!(
            validate_model_bytes(path, b"not an rknn model"),
            Err(Error::InvalidModel { .. })
        ));
        assert!(validate_model_bytes(path, b"RKNN\0\0\0\0\0\0\0\0\0\0\0\0").is_ok());
    }
}
