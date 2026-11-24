use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use foundationstereo_rknn::{F16Image, FoundationStereo, RknnOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let options = RknnOptions {
        library_path: args.get("rknn-lib").map(PathBuf::from),
        ..RknnOptions::default()
    };
    let mut stereo = FoundationStereo::load_with_options(required(&args, "model")?, options)?;
    let (width, height) = stereo.input_size();
    let left = read_f16_image(Path::new(required(&args, "left")?), height, width)?;
    let right = read_f16_image(Path::new(required(&args, "right")?), height, width)?;
    let disparity = stereo.infer(&left, &right)?;
    let output = Path::new(required(&args, "output")?);
    write_f32(&disparity, output)?;
    println!(
        "wrote {} shape={:?} values={}",
        output.display(),
        disparity.shape,
        disparity.data.len()
    );
    Ok(())
}

fn read_f16_image(
    path: &Path,
    height: usize,
    width: usize,
) -> Result<F16Image, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let expected = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .and_then(|values| values.checked_mul(2))
        .ok_or("image dimensions overflow usize")?;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected} for {width}x{height} NHWC FP16",
            path.display(),
            bytes.len()
        )
        .into());
    }
    let data = bytes
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes([word[0], word[1]]))
        .collect();
    Ok(F16Image {
        height,
        width,
        data,
    })
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

fn write_f32(tensor: &foundationstereo_rknn::Tensor, path: &Path) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(tensor.data.len() * 4);
    for value in &tensor.data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)
}
