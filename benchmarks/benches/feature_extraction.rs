use divan::{black_box, Bencher};
use feature_extraction::{akaze_keypoint_descriptor_extraction_def, export_matches, get_knn_matches, get_mat_from_dir, get_points_from_matches, ExtractedKeyPoint};
use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use opencv::core::{DMatch, KeyPoint, Vec3b, VecN, Vector};
use opencv::prelude::*;
use opencv::Error;
use std::env;
use std::sync::{Arc, Mutex};
use std::{io::Cursor, path::Path};
use std::sync::RwLock;

use homographier::homographier::{Cmat, MatError};

fn main() {
    divan::main();
}

/* 
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
            None
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
            .unwrap(),
            None
        )
        .unwrap()
        .0
        .len()
    );
}
*/

// Exponential increase the amount of keypoints for both images.
//     Reach the limit if viable.
// Increase resolution of only one image.
//     One image should be at a constant 25 keypoints while the other increases as before, with every test.

// QUICK NOTE: changed u32 to i32 because of akaze function
#[divan::bench(args = (1..19).map(|step| 2_i32.pow(step)), sample_size = 1, sample_count = 10)]
fn matching_test(bencher: Bencher, num_keypoints: i32) {
    // test image 1 & 2 
    let img1_dir = "../resources/test/benchmark/Denmark_8192-2.png";
    let img2_dir = "../resources/test/benchmark/Denmark_8192-2.png";

    // create image into Mat type
    let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
    let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

    // extract keypoints and descriptors from image 1 & 2
    let img1_keypoints = akaze_keypoint_descriptor_extraction_def(&img1, Some(num_keypoints)).unwrap();
    let img2_keypoints = akaze_keypoint_descriptor_extraction_def(&img2, Some(num_keypoints)).unwrap();
    
    // make mat into cmat (function from homo mod.rs called from_mat())
    // Then for each row and col for descriptors run at_2d() and use type u8
    // Converting Mat type into Cmat - this is for later use in the benchmark due to '*mut c_void' not being thread safe
    let descriptor_cmat1: Result<Cmat<u8>, MatError> = Cmat::from_mat(img1_keypoints.descriptors.clone());
    let descriptor_cmat2: Result<Cmat<u8>, MatError> = Cmat::from_mat(img2_keypoints.descriptors.clone());

    let mut all_descriptors1: Vec<Vec<u8>> = Vec::new();
    let mut all_descriptors2: Vec<Vec<u8>> = Vec::new();

    // Going through every descriptor and adding them to to all_descriptors as Vec<u8>
    match descriptor_cmat1 {
        Ok(Cmat) => {
                for row in 0..2 {
                    let mut row_descriptor1: Vec<u8> = Vec::new();
                    for col in 0..61 {
                        match Cmat.at_2d(row, col) {
                            Ok(value) => row_descriptor1.push(*value),
                            Err(_) => {}
                        }
                    }
                    all_descriptors1.push(row_descriptor1);
                }
        },
        Err(e) => println!("Failed to convert Mat to Cmat<Vec<u8>> {:?}", e),
    }

    // Going through every descriptor and adding them to to all_descriptors as Vec<u8>
    match descriptor_cmat2 {
        Ok(Cmat) => {
                for row in 0..2 {
                    let mut row_descriptor2: Vec<u8> = Vec::new();
                    for col in 0..61 {
                        match Cmat.at_2d(row, col) {
                            Ok(value) => row_descriptor2.push(*value),
                            Err(_) => {}
                        }
                    }
                    all_descriptors2.push(row_descriptor2);
                }
        },
        Err(e) => println!("Failed to convert Mat to Cmat<Vec<u8>> {:?}", e),
    }

    // The benchmark
    bencher.with_inputs(|| -> (Mat, Mat) {
        // Convert Vec<Vec<u8>> to Vec<&[u8]>, because from_slice_2d needs a slice
        let slice_of_slices1: Vec<&[u8]> = all_descriptors1.iter().map(AsRef::as_ref).collect();
        let slice_of_slices2: Vec<&[u8]> = all_descriptors2.iter().map(AsRef::as_ref).collect();

        // Locally creating Mat from slice of slices 
        (
            Mat::from_slice_2d(&slice_of_slices1)
                .expect("Failed to reconstruct Mat"), 
            Mat::from_slice_2d(&slice_of_slices2)
                .expect("Failed to reconstruct Mat")
        )
    })
    .bench_refs(|(reconstructed_mat1, reconstructed_mat2)| {
        {
            get_knn_matches(
                &reconstructed_mat1, 
                &reconstructed_mat2, 
                2, 
                0.3)
        }

    });

    println!(
        "Matches: {}, Keypoints: img1({}) img2({})",
        get_knn_matches(&img1_keypoints.descriptors, &img2_keypoints.descriptors, 2, 0.3).unwrap().len(),
        img1_keypoints.keypoints.len(),
        img2_keypoints.keypoints.len()
    );

}