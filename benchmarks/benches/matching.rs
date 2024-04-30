use cv::{
    core::{DMatch, KeyPoint, Mat, Point2f, Vector, NORM_HAMMING},
    features2d::{AKAZE_DescriptorType, BFMatcher, DrawMatchesFlags, KAZE_DiffusivityType, AKAZE},
    imgcodecs,
    types::{VectorOfDMatch, VectorOfPoint2f, VectorOfVectorOfDMatch},
    Error,
};
use feature_extraction::{akaze_keypoint_descriptor_extraction_def, get_mat_from_dir};
use opencv::core::Ptr;
use opencv::{self as cv, prelude::*};
use divan::{black_box, Bencher};


/// Matching test:
/// Want to test how many matches relative to the keypoints
/// 

fn matching_test() {
    // test image 1 & 2
    let img1_dir = "../resources/test/Geotiff/30.tif";
    let img2_dir = "../resources/test/Geotiff/31.tif";

    // create image into Mat type
    let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
    let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

    // extract keypoints from image 1 & 2
    let img1_keypoints = akaze_keypoint_descriptor_extraction_def(&img1).unwrap();
    let img2_keypoints = akaze_keypoint_descriptor_extraction_def(&img2).unwrap();
}