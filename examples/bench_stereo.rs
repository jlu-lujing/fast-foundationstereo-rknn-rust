use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Instant,
};

use foundationstereo_rknn::{F16Image, FoundationStereo, NpuCore, RknnOptions, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let options = RknnOptions {
        library_path: args.get("rknn-lib").map(PathBuf::from),
        core: args
            .get("core")
            .map(|value| parse_core(value))
            .transpose()?
            .unwrap_or(NpuCore::All),
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

    let model = measure_tensor(frames, warmup, || {
        stereo.infer(&prepared_left, &prepared_right)
    })?;
    let mut reused_output = vec![0.0; stereo.output_len()];
    let model_into = measure_into(frames, warmup, &mut reused_output, |output| {
        stereo.infer_into(&prepared_left, &prepared_right, output)
    })?;
    let rgb8 = measure_tensor(frames, warmup, || stereo.infer_rgb8(&left, &right))?;
    let rgb8_into = measure_into(frames, warmup, &mut reused_output, |output| {
        stereo.infer_rgb8_into(&left, &right, output)
    })?;
    println!(
        "size={width}x{height} frames={frames} model={model} model_into={model_into} \
         rgb8={rgb8} rgb8_into={rgb8_into}"
    );

    if let Some(path) = args.get("output") {
        let disparity = stereo.infer_rgb8(&left, &right)?;
        let mut reused = vec![0.0; stereo.output_len()];
        stereo.infer_rgb8_into(&left, &right, &mut reused)?;
        if disparity.data != reused {
            return Err("preallocated and owned output paths disagree".into());
        }
        write_f32(&disparity, Path::new(path))?;
        println!("wrote {path} (owned and preallocated outputs match)");
    }
    Ok(())
}

fn measure_tensor(
    frames: usize,
    warmup: usize,
    mut infer: impl FnMut() -> foundationstereo_rknn::Result<Tensor>,
) -> Result<String, Box<dyn std::error::Error>> {
    measure_unit(frames, warmup, || infer().map(drop))
}

fn measure_into(
    frames: usize,
    warmup: usize,
    output: &mut [f32],
    mut infer: impl FnMut(&mut [f32]) -> foundationstereo_rknn::Result<()>,
) -> Result<String, Box<dyn std::error::Error>> {
    measure_unit(frames, warmup, || infer(output))
}

fn measure_unit(
    frames: usize,
    warmup: usize,
    mut infer: impl FnMut() -> foundationstereo_rknn::Result<()>,
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

fn parse_core(value: &str) -> Result<NpuCore, Box<dyn std::error::Error>> {
    match value {
        "auto" => Ok(NpuCore::Auto),
        "0" => Ok(NpuCore::Core0),
        "1" => Ok(NpuCore::Core1),
        "2" => Ok(NpuCore::Core2),
        "01" => Ok(NpuCore::Core01),
        "7" | "all" => Ok(NpuCore::All),
        _ => Err(format!("invalid --core {value:?}; expected auto, 0, 1, 2, 01, or 7").into()),
    }
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
