use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use foundationstereo_rknn::{
    NpuCore, RknnModel, RknnOptions, Tensor,
    cpu::{OrtCpuStages, OrtOptions},
    gwc::{GwcBackend, ScalarGwc},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let artifacts = PathBuf::from(required(&args, "artifacts")?);
    let rknn_library = args.get("rknn-lib").map(PathBuf::from);
    let output = PathBuf::from(required(&args, "out")?);
    std::fs::create_dir_all(&output)?;
    let left = load_image_nchw(Path::new(required(&args, "left")?))?;
    let right = load_image_nchw(Path::new(required(&args, "right")?))?;
    let load = |name: &str, core| {
        RknnModel::load_with_options(
            artifacts.join(name),
            RknnOptions {
                library_path: rknn_library.clone(),
                core,
            },
        )
    };
    let mut feature = load("feature_single.rknn", NpuCore::Core1)?;
    let mut pre = load("segment_pre_resize.rknn", NpuCore::Core01)?;
    let mut post = load("segment_post_resize_full.rknn", NpuCore::Core2)?;
    let left_features = feature.run(&[left])?;
    let right_features = feature.run(&[right])?;
    let left_04 = left_features[0].clone();
    let right_04 = right_features[0].clone();
    let stem = left_features[4].clone();
    let volume = ScalarGwc.volume(&left_04, &right_04, 8, 24)?;
    let pre_outputs = pre.run(&[
        left_04.clone(),
        left_features[1].clone(),
        left_features[2].clone(),
        left_features[3].clone(),
        right_04.clone(),
        volume,
    ])?;
    write_tensor(&output.join("pre_attention.f32"), &pre_outputs[0])?;
    write_tensor(&output.join("pre_corr.f32"), &pre_outputs[1])?;
    let refine_model = args
        .get("refine")
        .map(PathBuf::from)
        .unwrap_or_else(|| artifacts.join("segment_refine.onnx"));
    let mut cpu = OrtCpuStages::load(
        artifacts.join("stage_resize_only.onnx"),
        refine_model,
        &OrtOptions {
            dylib_path: args.get("ort").map(PathBuf::from),
            refine_operator_library: args.get("ffs-lib").map(PathBuf::from),
            resize_threads: 1,
            refine_threads: 4,
        },
    )?;
    let resized = cpu.resize(&pre_outputs[0])?;
    write_tensor(&output.join("resized.f32"), &resized)?;
    let post_outputs = post.run(&[resized, pre_outputs[1].clone()])?;
    write_tensor(&output.join("p8.f32"), &post_outputs[0])?;
    write_tensor(&output.join("initial.f32"), &post_outputs[1])?;
    let disparity = cpu.refine(
        &post_outputs[1],
        &post_outputs[0],
        &left_04,
        &right_04,
        &stem,
    )?;
    write_tensor(&output.join("disparity.f32"), &disparity)?;
    Ok(())
}

fn write_tensor(path: &Path, tensor: &Tensor) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(tensor.data.len() * 4);
    for value in &tensor.data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    println!("{} {:?}", path.display(), tensor.shape);
    Ok(())
}

fn parse_args() -> HashMap<String, String> {
    let mut values = HashMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if let Some(key) = flag.strip_prefix("--")
            && let Some(value) = args.next()
        {
            values.insert(key.to_string(), value);
        }
    }
    values
}

fn required<'a>(
    args: &'a HashMap<String, String>,
    key: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    args.get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("missing --{key}").into())
}

fn load_image_nchw(path: &Path) -> Result<Tensor, Box<dyn std::error::Error>> {
    let image = image::open(path)?.to_rgb8();
    let (width, height) = image.dimensions();
    let mut data = vec![0.0f32; 3 * height as usize * width as usize];
    let plane = height as usize * width as usize;
    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = image.get_pixel(x as u32, y as u32).0;
            for channel in 0..3 {
                data[channel * plane + y * width as usize + x] = pixel[channel] as f32;
            }
        }
    }
    Ok(Tensor::new(
        vec![1, 3, height as usize, width as usize],
        data,
    )?)
}
