use std::{env, path::PathBuf};

use foundationstereo_rknn::{NpuCore, RknnModel, RknnOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env::args().nth(1).unwrap_or_else(|| ".".to_string()));
    let library = env::var_os("RKNN_LIBRARY").map(PathBuf::from);
    for (name, core) in [
        ("feature_single.rknn", NpuCore::Core1),
        ("segment_pre_resize.rknn", NpuCore::Core01),
        ("segment_post_resize_full.rknn", NpuCore::Core2),
    ] {
        let model = RknnModel::load_with_options(
            root.join(name),
            RknnOptions {
                library_path: library.clone(),
                core,
            },
        )?;
        println!("{name}");
        for input in model.input_info() {
            println!(
                "  input  {} {} {:?} ({} bytes)",
                input.index, input.name, input.shape, input.bytes
            );
        }
        for output in model.output_info() {
            println!(
                "  output {} {} {:?} ({} bytes)",
                output.index, output.name, output.shape, output.bytes
            );
        }
    }
    Ok(())
}
