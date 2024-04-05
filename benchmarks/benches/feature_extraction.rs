use divan::Bencher;
use std::{io::Cursor, path::Path};
use image::io::Reader as ImageReader;
use std::env;

fn main() {
    divan::main();
}

#[divan::bench(args = [128,256,512,1024, 2048, 4096, 8192])]
fn extract_features_from_image(bencher: Bencher, sample_size: i32) {
    let mut current_dir = env::current_dir().expect("Current directory not set.");
    current_dir.pop();
    current_dir.push("resources/test/benchmark/Denmark_16384.png");

    let test_image = ImageReader::open(current_dir).unwrap();
}