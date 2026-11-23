use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Instant,
};

use foundationstereo_rknn::{FoundationStereo, PipelineOptions, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let mut options = PipelineOptions::new(PathBuf::from(required(&args, "artifacts")?));
    options.rknn_library = args.get("rknn-lib").map(PathBuf::from);
    options.ort.dylib_path = args.get("ort").map(PathBuf::from);
    if let Some(library) = args.get("ffs-lib") {
        options.ort.refine_operator_library = Some(PathBuf::from(library));
        options.artifacts.prefer_ffs_refine();
    }
    options.ort.refine_threads = args
        .get("refine-threads")
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(4);
    let mut stereo = FoundationStereo::load(options)?;
    let left = load_image_nchw(Path::new(required(&args, "left")?))?;
    let right = load_image_nchw(Path::new(required(&args, "right")?))?;
    let warmup = args
        .get("warmup")
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(2usize);
    let frames = args
        .get("frames")
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(10usize);

    for _ in 0..warmup {
        stereo.infer(&left, &right)?;
    }
    let mut times = Vec::with_capacity(frames);
    for _ in 0..frames {
        let started = Instant::now();
        stereo.infer(&left, &right)?;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    let p95 = times[((times.len() as f64 * 0.95).ceil() as usize - 1).min(times.len() - 1)];
    println!(
        "frames={frames} median={median:.1} ms p95={p95:.1} ms fps={:.3}",
        1000.0 / median
    );
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
