fn main() {
    cc::Build::new()
        .file("csrc/gwc_volume_neon.c")
        .opt_level(3)
        .warnings(true)
        .compile("ffs_gwc");
    println!("cargo:rerun-if-changed=csrc/gwc_volume_neon.c");
}
