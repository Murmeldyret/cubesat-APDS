use divan::{black_box, Bencher};
use feature_extraction::akaze_keypoint_descriptor_extraction_def;
use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use opencv::core::{KeyPoint, Vec3b, VecN, Vector};
use opencv::prelude::*;
use opencv::Error;
use std::env;
use std::{io::Cursor, path::Path};

fn main() {
    divan::main();
}

#[divan::bench(args = (1..13).map(|step| 2_u32.pow(step)), sample_size = 1, sample_count = 1)]
fn extract_features_from_image(bencher: Bencher, sample_size: u32) {
    let mut current_dir = env::current_dir().expect("Current directory not set.");
    current_dir.pop();
    current_dir.push("resources/test/benchmark/Denmark_8192.png");

    let test_image = ImageReader::open(current_dir)
        .unwrap()
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();

    let resized_image = test_image.resize(sample_size, sample_size, FilterType::Lanczos3);

    let image_vec: Vec<VecN<u8, 3>> = resized_image
        .into_rgb8()
        .pixels()
        .map(|pixel| VecN::from_array(pixel.0))
        .collect();

    bencher.bench(|| {
        akaze_keypoint_descriptor_extraction_def(
            &Mat::from_slice_rows_cols::<Vec3b>(
                &image_vec,
                sample_size as usize,
                sample_size as usize,
            )
            .unwrap(),
        )
    });
    println!(
        "\nKeypoints detected: {}",
        akaze_keypoint_descriptor_extraction_def(
            &Mat::from_slice_rows_cols::<Vec3b>(
                &image_vec,
                sample_size as usize,
                sample_size as usize
            )
            .unwrap()
        )
        .unwrap()
        .0
        .len()
    );
}
