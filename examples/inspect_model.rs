use std::{env, path::PathBuf};

use foundationstereo_rknn::{NpuCore, RknnModel, RknnOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = PathBuf::from(
        env::args()
            .nth(1)
            .unwrap_or_else(|| "stereo_192x384_it2_d96.rknn".to_string()),
    );
    let library_path = env::var_os("RKNN_LIBRARY").map(PathBuf::from);
    let model = RknnModel::load_with_options(
        model_path,
        RknnOptions {
            library_path,
            core: NpuCore::All,
        },
    )?;
    for input in model.input_info() {
        println!(
            "input  {} {} {:?} ({} bytes)",
            input.index, input.name, input.shape, input.bytes
        );
    }
    for output in model.output_info() {
        println!(
            "output {} {} {:?} ({} bytes)",
            output.index, output.name, output.shape, output.bytes
        );
    }
    Ok(())
}
