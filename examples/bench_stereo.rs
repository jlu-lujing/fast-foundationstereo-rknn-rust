use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Instant,
};

use foundationstereo_rknn::{F16Image, FoundationStereo, RknnOptions, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let options = RknnOptions {
        library_path: args.get("rknn-lib").map(PathBuf::from),
        ..RknnOptions::default()
    };
    let mut stereo = FoundationStereo::load_with_options(required(&args, "model")?, options)?;
    let (width, height) = stereo.input_size();
    let left = load_resized_rgb(Path::new(required(&args, "left")?), width, height)?;
    let right = load_resized_rgb(Path::new(required(&args, "right")?), width, height)?;
    let prepared_left = F16Image::from_rgb8(height, width, &left)?;
    let prepared_right = F16Image::from_rgb8(height, width, &right)?;
    let warmup = parse_usize(&args, "warmup", 5)?;
    let frames = parse_usize(&args, "frames", 50)?;
    if frames == 0 {
        return Err("--frames must be positive".into());
    }

    let model = measure(frames, warmup, || {
        stereo.infer(&prepared_left, &prepared_right)
    })?;
    let rgb8 = measure(frames, warmup, || stereo.infer_rgb8(&left, &right))?;
    println!("size={width}x{height} frames={frames} model={model} rgb8={rgb8}");

    if let Some(path) = args.get("output") {
        let disparity = stereo.infer_rgb8(&left, &right)?;
        write_f32(&disparity, Path::new(path))?;
        println!("wrote {path}");
    }
    Ok(())
}

fn measure(
    frames: usize,
    warmup: usize,
    mut infer: impl FnMut() -> foundationstereo_rknn::Result<Tensor>,
) -> Result<String, Box<dyn std::error::Error>> {
    for _ in 0..warmup {
        infer()?;
    }
    let mut times = Vec::with_capacity(frames);
    for _ in 0..frames {
        let started = Instant::now();
        infer()?;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    let p95 = times[((times.len() as f64 * 0.95).ceil() as usize - 1).min(times.len() - 1)];
    Ok(format!(
        "median={median:.3}ms p95={p95:.3}ms fps={:.3}",
        1000.0 / median
    ))
}

fn parse_usize(
    args: &HashMap<String, String>,
    key: &str,
    default: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    args.get(key)
        .map(|value| value.parse().map_err(Into::into))
        .unwrap_or(Ok(default))
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

fn load_resized_rgb(
    path: &Path,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let image = image::open(path)?.to_rgb8();
    let resized = image::imageops::resize(
        &image,
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    Ok(resized.into_raw())
}

fn write_f32(tensor: &Tensor, path: &Path) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(tensor.data.len() * 4);
    for value in &tensor.data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)
}
