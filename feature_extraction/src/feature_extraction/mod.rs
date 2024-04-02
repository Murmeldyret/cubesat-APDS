use anyhow::anyhow;
use anyhow::Context;
use anyhow::Result;
use cv::core::NORM_HAMMING;
//use cv::core::NORM_L2;
use cv::{
    core::{DMatch, InputArray, Vector, NORM_L2},
    features2d::{DescriptorMatcher, DescriptorMatcher_FLANNBASED},
    types,
};
use image::RgbImage;
use ndarray::{Array1, ArrayView1, ArrayView3};
use opencv::{self as cv, prelude::*};
fn main() -> Result<()> { 
    // Read image
    let img = cv::imgcodecs::imread("./Data/9.png", cv::imgcodecs::IMREAD_COLOR)?;
    let img2 = cv::imgcodecs::imread("./Data/10.png", cv::imgcodecs::IMREAD_COLOR)?;
    // Use Orb
    let mut orba = <cv::features2d::ORB>::create(
        50,
        1.2,
        8,
        31,
        0,
        2,
        cv::features2d::ORB_ScoreType::HARRIS_SCORE,
        31,
        20,
    )?;

    let mut orb = <cv::features2d::AKAZE>::create(
        cv::features2d::AKAZE_DescriptorType::DESCRIPTOR_MLDB,
        0,
        3,
        0.001f32,
        4,
        4,
        opencv::features2d::KAZE_DiffusivityType::DIFF_PM_G2,
        10,
    )?;

    let mut orb_keypoints = cv::core::Vector::default();
    let mut orb_desc = cv::core::Mat::default();
    let mut dst_img = cv::core::Mat::default();
    let mask = cv::core::Mat::default();

    let mut orb_keypoints2 = cv::core::Vector::default();
    let mut orb_desc2 = cv::core::Mat::default();
    let mut dst_img2 = cv::core::Mat::default();
    let mask2 = cv::core::Mat::default();

    orb.detect_and_compute(&img, &mask, &mut orb_keypoints, &mut orb_desc, false)?;
    cv::features2d::draw_keypoints(
        &img,
        &orb_keypoints,
        &mut dst_img,
        cv::core::VecN([0., 255., 0., 255.]),
        cv::features2d::DrawMatchesFlags::DEFAULT,
    )?;

    orb.detect_and_compute(&img2, &mask2, &mut orb_keypoints2, &mut orb_desc2, false)?;
    cv::features2d::draw_keypoints(
        &img2,
        &orb_keypoints2,
        &mut dst_img2,
        cv::core::VecN([0., 255., 0., 255.]),
        cv::features2d::DrawMatchesFlags::DEFAULT,
    )?;

    cv::imgproc::rectangle(
        &mut dst_img,
        cv::core::Rect::from_points(cv::core::Point::new(0, 0), cv::core::Point::new(50, 50)),
        cv::core::VecN([255., 0., 0., 0.]),
        -1,
        cv::imgproc::LINE_8,
        0,
    )?;

    cv::imgproc::rectangle(
        &mut dst_img2,
        cv::core::Rect::from_points(cv::core::Point::new(0, 0), cv::core::Point::new(50, 50)),
        cv::core::VecN([255., 0., 0., 0.]),
        -1,
        cv::imgproc::LINE_8,
        0,
    )?;

    /*
    // Use SIFT
    let mut sift = cv::features2d::SIFT::create(0, 3, 0.04, 10., 1.6, false)?;
    let mut sift_keypoints = cv::core::Vector::default();
    let mut sift_desc = cv::core::Mat::default();
    sift.detect_and_compute(&img, &mask, &mut sift_keypoints, &mut sift_desc, false)?;
    
    cv::features2d::draw_keypoints(
        &dst_img.clone(),
        &sift_keypoints,
        &mut dst_img,
        cv::core::VecN([0., 0., 255., 255.]),
        cv::features2d::DrawMatchesFlags::DEFAULT,
    )?;
    */

    //let matcher = opencv::features2d::BFMatcher::create(NORM_L2, false);
    //let desc_matcher = opencv::features2d::DescriptorMatcher::create_with_matcher_type(opencv::features2d::DescriptorMatcher_MatcherType::BRUTEFORCE);

    

    let mut matches = opencv::types::VectorOfDMatch::new();
    let mut bf_matcher = cv::features2d::BFMatcher::new(cv::core::NORM_L1, true)?;

    let mut flann_matcher = cv::features2d::FlannBasedMatcher::new_def()?;

    // The matching algorithm
	bf_matcher.train_match_def(&orb_desc, &orb_desc2, &mut matches).unwrap();

    println!("Matches: {:#?}", matches);

    //let matches1to2 = opencv::core::Vector::new();
    let mut out_img = cv::core::Mat::default();
    let matches_mask = cv::core::Vector::new();

    cv::features2d::draw_matches(
        &img,
        &orb_keypoints,
        &img2,
        &orb_keypoints2,
        &matches,
        &mut out_img,
        opencv::core::VecN::all(-1.0),
        opencv::core::VecN::all(-1.0),
        &matches_mask,
        cv::features2d::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
     )?;
   


    // Write image using OpenCV
    cv::imgcodecs::imwrite("./tmp.png", &dst_img, &cv::core::Vector::default())?;
    cv::imgcodecs::imwrite("./tmp2.png", &dst_img2, &cv::core::Vector::default())?;

    cv::imgcodecs::imwrite("./tmp3.png", &out_img, &cv::core::Vector::default())?;

    // Convert :: cv::core::Mat -> ndarray::ArrayView3
    let a = dst_img.try_as_array()?;
    // Convert :: ndarray::ArrayView3 -> RgbImage
    // Note, this require copy as RgbImage will own the data
    let test_image = array_to_image(a);
    // Note, the colors will be swapped (BGR <-> RGB)
  	// Will need to swap the channels before
    // converting to RGBImage
    // But since this is only a demo that
    // it indeed works to convert cv::core::Mat -> ndarray::ArrayView3
    // I'll let it be
    test_image.save("out.png")?;
Ok(())
}
trait AsArray {
    fn try_as_array(&self) -> Result<ArrayView3<u8>>;
}
impl AsArray for cv::core::Mat {
    fn try_as_array(&self) -> Result<ArrayView3<u8>> {
        if !self.is_continuous() {
            return Err(anyhow!("Mat is not continuous"));
        }
        let bytes = self.data_bytes()?;
        let size = self.size()?;
        let a = ArrayView3::from_shape((size.height as usize, size.width as usize, 3), bytes)?;
        Ok(a)
    }
}
// From Stack Overflow: https://stackoverflow.com/questions/56762026/how-to-save-ndarray-in-rust-as-image
fn array_to_image(arr: ArrayView3<u8>) -> RgbImage {
    assert!(arr.is_standard_layout());
let (height, width, _) = arr.dim();
    let raw = arr.to_slice().expect("Failed to extract slice from array");
RgbImage::from_raw(width as u32, height as u32, raw.to_vec())
        .expect("container should have the right size for the image dimensions")
}