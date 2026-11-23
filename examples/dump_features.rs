use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use foundationstereo_rknn::{
    NpuCore, RknnModel, RknnOptions, Tensor,
    gwc::{GwcBackend, ScalarGwc},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let artifacts = PathBuf::from(required(&args, "artifacts")?);
    let left = load_image_nchw(Path::new(required(&args, "left")?))?;
    let right = load_image_nchw(Path::new(required(&args, "right")?))?;
    let output = PathBuf::from(required(&args, "out")?);
    std::fs::create_dir_all(&output)?;

    let mut encoder = RknnModel::load_with_options(
        artifacts.join("feature_single.rknn"),
        RknnOptions {
            library_path: args.get("rknn-lib").map(PathBuf::from),
            core: NpuCore::Core1,
        },
    )?;
    let left_outputs = encoder.run(&[left])?;
    let right_outputs = encoder.run(&[right])?;
    for (prefix, outputs) in [("left", &left_outputs), ("right", &right_outputs)] {
        for (index, tensor) in outputs.iter().enumerate() {
            write_tensor(
                &output.join(format!("{prefix}_{index}_{}.f32", tensor.name_or("out"))),
                tensor,
            )?;
        }
    }
    let volume = ScalarGwc.volume(&left_outputs[0], &right_outputs[0], 8, 24)?;
    write_tensor(&output.join("volume.f32"), &volume)?;
    println!("{:?}", volume.shape);
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
