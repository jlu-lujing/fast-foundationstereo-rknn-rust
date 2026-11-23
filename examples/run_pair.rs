use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use foundationstereo_rknn::{FoundationStereo, PipelineOptions, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let artifacts = required(&args, "artifacts")?;
    let left_path = PathBuf::from(required(&args, "left")?);
    let right_path = PathBuf::from(required(&args, "right")?);

    let mut options = PipelineOptions::new(artifacts);
    if let Some(refine) = args.get("refine") {
        options.artifacts.refine_onnx = PathBuf::from(refine);
    }
    options.rknn_library = args.get("rknn-lib").map(PathBuf::from);
    options.ort.dylib_path = args.get("ort").map(PathBuf::from);
    if let Some(library) = args.get("ffs-lib") {
        options.ort.refine_operator_library = Some(PathBuf::from(library));
        options.artifacts.prefer_ffs_refine();
    }
    if let Some(threads) = args.get("refine-threads") {
        options.ort.refine_threads = threads.parse()?;
    }
    let mut stereo = FoundationStereo::load(options)?;
    let left = load_image_nchw(&left_path)?;
    let right = load_image_nchw(&right_path)?;
    let trace = stereo.infer_trace(&left, &right)?;
    let disparity = &trace.disparity;
    if let Some(directory) = args.get("trace-dir") {
        let directory = Path::new(directory);
        std::fs::create_dir_all(directory)?;
        write_tensor(&directory.join("volume.f32"), &trace.volume)?;
        write_tensor(&directory.join("pre_attention.f32"), &trace.pre_outputs[0])?;
        write_tensor(&directory.join("pre_corr.f32"), &trace.pre_outputs[1])?;
        write_tensor(&directory.join("resized.f32"), &trace.resized)?;
        write_tensor(&directory.join("p8.f32"), &trace.post_outputs[0])?;
        write_tensor(&directory.join("initial.f32"), &trace.post_outputs[1])?;
        write_tensor(&directory.join("disparity.f32"), disparity)?;
    }

    if let Some(output) = args.get("output") {
        save_disparity(disparity, Path::new(output))?;
        println!("wrote {output}");
    }
    if let Some(output) = args.get("raw-output") {
        save_raw_disparity(disparity, Path::new(output))?;
        println!("wrote {output}");
    }
    println!(
        "disparity shape={:?} min={:.3} max={:.3} mean={:.3}",
        disparity.shape,
        disparity.data.iter().copied().fold(f32::INFINITY, f32::min),
        disparity
            .data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
        disparity.data.iter().sum::<f32>() / disparity.data.len() as f32
    );
    Ok(())
}

fn write_tensor(path: &Path, tensor: &Tensor) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(tensor.data.len() * 4);
    for value in &tensor.data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)
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

fn save_disparity(disparity: &Tensor, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (height, width) = match disparity.shape.as_slice() {
        [1, 1, height, width] => (*height as u32, *width as u32),
        _ => return Err("expected disparity shape [1,1,H,W]".into()),
    };
    let maximum = disparity
        .data
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0f32, f32::max)
        .max(1.0);
    let mut image = image::GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let value = disparity.data[(y * width + x) as usize];
            let scaled = ((value / maximum) * 255.0).clamp(0.0, 255.0) as u8;
            image.put_pixel(x, y, image::Luma([scaled]));
        }
    }
    image.save(path)?;
    Ok(())
}

fn save_raw_disparity(disparity: &Tensor, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(disparity.data.len() * 4);
    for value in &disparity.data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}
