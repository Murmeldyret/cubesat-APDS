use anyhow::anyhow;
use anyhow::Result;
use cv::{
    core::{DMatch, InputArray, Vector},
    features2d::{DescriptorMatcher, AKAZE}
};
use opencv::core::Ptr;

use opencv::{self as cv, prelude::*};

#[derive(Default)]
pub struct AkazeStruct {
    pub scaling: Option<(usize, usize)>,
    pub red_band_index: Option<isize>,
    pub green_band_index: Option<isize>,
    pub blue_band_index: Option<isize>,
}

pub fn akaze_keypoint_extraction_def(file_location: &str) -> Vector<cv::core::KeyPoint> {
    let img: Mat = cv::imgcodecs::imread(file_location, cv::imgcodecs::IMREAD_COLOR).unwrap();

    let mut akaze: Ptr<AKAZE> = <cv::features2d::AKAZE>::create(
        cv::features2d::AKAZE_DescriptorType::DESCRIPTOR_MLDB,
        0,
        3,
        0.001f32,
        4,
        4,
        opencv::features2d::KAZE_DiffusivityType::DIFF_PM_G2,
        -1,
    ).unwrap();

    let mut akaze_keypoints = cv::core::Vector::default();
    let mut akaze_desc = cv::core::Mat::default();
    let mut dst_img = cv::core::Mat::default();
    let mask = cv::core::Mat::default();

    akaze.detect_and_compute(&img, &mask, &mut akaze_keypoints, &mut akaze_desc, false).unwrap();
    cv::features2d::draw_keypoints(
        &img,
        &akaze_keypoints,
        &mut dst_img,
        cv::core::VecN([0., 255., 0., 255.]),
        cv::features2d::DrawMatchesFlags::DEFAULT,
    ).unwrap();

    cv::imgcodecs::imwrite("./tmp.png", &dst_img, &cv::core::Vector::default()).unwrap();

    return akaze_keypoints;  
}

#[allow(clippy::unwrap_used)]
#[allow(unused_variables)]
#[allow(unused_imports)]
#[allow(dead_code)]
mod test {
    use opencv::{
        core::*,
        imgcodecs::{ImreadModes, IMREAD_UNCHANGED},
    };

    use std::env;

    use super::akaze_keypoint_extraction_def;

    #[test]
    fn test_test() {
        let path = env::current_dir().unwrap();
        println!("The current directory is {}", path.display());
        akaze_keypoint_extraction_def("./1.png");
        assert!(true);
    }
}
