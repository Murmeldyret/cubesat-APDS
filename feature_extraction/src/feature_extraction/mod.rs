use anyhow::anyhow;
use anyhow::Result;
use cv::{
    core::{DMatch, InputArray, Vector, KeyPoint},
    features2d::{DescriptorMatcher, AKAZE},
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

pub fn akaze_keypoint_descriptor_extraction_def(img: &Mat) -> (Vector<KeyPoint>, cv::core::Mat) {
    //let img: Mat = cv::imgcodecs::imread(file_location, cv::imgcodecs::IMREAD_COLOR).unwrap();

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

    return (akaze_keypoints, akaze_desc);  
}

pub fn get_bruteforce_matches(origin_desc: cv::core::Mat, target_desc: cv::core::Mat) -> Vector<DMatch> {
    let mut matches= opencv::types::VectorOfVectorOfDMatch::new();
    let mut bf_matcher = cv::features2d::BFMatcher::new(cv::core::NORM_HAMMING, false).unwrap();

    bf_matcher.knn_train_match_def(&origin_desc, &target_desc, &mut matches, 2).unwrap();

    let mut good_matches = opencv::types::VectorOfDMatch::new();
    let mut tracker = 0i32;

    for i in &matches {
        for m in &i {
            for n in &i {
                tracker += 1;
                if m.distance < 0.3 * n.distance {
                    good_matches.push(m);
                    break
                }
            }
        }
    }

    return good_matches;
}

pub fn export_matches(
    img1: &Mat,
    img1_keypoints: &Vector<KeyPoint>,
    img2: &Mat,
    img2_keypoints: &Vector<KeyPoint>,
    matches: &Vector<DMatch>,
    export_location: &str
) {

    let mut out_img = cv::core::Mat::default();
    let matches_mask = cv::core::Vector::new();

    cv::features2d::draw_matches(
        &img1,
        &img1_keypoints,
        &img2,
        &img2_keypoints,
        &matches,
        &mut out_img,
        opencv::core::VecN::all(-1.0),
        opencv::core::VecN::all(-1.0),
        &matches_mask,
        cv::features2d::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
     ).unwrap();

     cv::imgcodecs::imwrite(export_location, &out_img, &cv::core::Vector::default()).unwrap();
}

#[allow(clippy::unwrap_used)]
#[allow(unused_variables)]
#[allow(unused_imports)]
#[allow(dead_code)]
mod test {

    use opencv::{self as cv, prelude::*};

    use std::env;

    use crate::feature_extraction::{export_matches, get_bruteforce_matches};

    use super::akaze_keypoint_descriptor_extraction_def;

    #[test]
    fn fake_test() {
        let path = env::current_dir().unwrap();
        println!("The current directory is {}", path.display());

        let img1_location = "./30.tif";
        let img2_location = "./31.tif";

        let img1: Mat = cv::imgcodecs::imread(img1_location, cv::imgcodecs::IMREAD_COLOR).unwrap();
        let img2: Mat = cv::imgcodecs::imread(img2_location, cv::imgcodecs::IMREAD_COLOR).unwrap();

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1);
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2);
        
        println!("{} - Keypoints: {}", img1_location, img1_keypoints.len());
        println!("{} - Keypoints: {}", img2_location, img2_keypoints.len());

        let matches = get_bruteforce_matches(img1_desc, img2_desc);

        println!("Matches: {}", matches.len());

        export_matches(&img1, &img1_keypoints, &img2, &img2_keypoints, &matches, "./out.tif", );

        assert!(true);
    }
}
