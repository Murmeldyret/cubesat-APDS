use cv::{
    core::{DMatch, Vector, KeyPoint},
    features2d::AKAZE,
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

    return (akaze_keypoints, akaze_desc);  
}

pub fn get_knn_matches(origin_desc: cv::core::Mat, target_desc: cv::core::Mat, k: i32, filter_strength: f32) -> Vector<DMatch> {
    let mut matches= opencv::types::VectorOfVectorOfDMatch::new();
    let bf_matcher = cv::features2d::BFMatcher::new(cv::core::NORM_HAMMING, false).unwrap();

    bf_matcher.knn_train_match_def(&origin_desc, &target_desc, &mut matches, k).unwrap();

    let mut good_matches = opencv::types::VectorOfDMatch::new();

    for i in &matches {
        for m in &i {
            for n in &i {
                if m.distance < filter_strength * n.distance {
                    good_matches.push(m);
                    break
                }
            }
        }
    }

    return good_matches;
}

pub fn get_bruteforce_matches(origin_desc: cv::core::Mat, target_desc: cv::core::Mat) -> Vector<DMatch> {
    let mut matches = opencv::types::VectorOfDMatch::new();
    let bf_matcher = cv::features2d::BFMatcher::new(cv::core::NORM_HAMMING, true).unwrap();

    bf_matcher.train_match_def(&origin_desc, &target_desc, &mut matches).unwrap();

    return matches;
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

pub fn get_mat_from_dir(img_dir: &str) -> Mat {
    return cv::imgcodecs::imread(img_dir, cv::imgcodecs::IMREAD_COLOR).unwrap();
}

#[allow(clippy::unwrap_used)]
#[allow(unused_variables)]
#[allow(unused_imports)]
#[allow(dead_code)]
mod test {

    use opencv::{self as cv, prelude::*};

    use crate::feature_extraction::{export_matches, get_knn_matches, get_mat_from_dir, get_bruteforce_matches};

    use super::akaze_keypoint_descriptor_extraction_def;

    #[test]
    fn fake_test() {
        let img1_dir = "./30.tif";
        let img2_dir = "./31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir);
        let img2: Mat = get_mat_from_dir(img2_dir);

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1);
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2);
        
        println!("{} - Keypoints: {}", img1_dir, img1_keypoints.len());
        println!("{} - Keypoints: {}", img2_dir, img2_keypoints.len());

        let matches = get_knn_matches(img1_desc, img2_desc, 2, 0.3);

        println!("Matches: {}", matches.len());

        export_matches(&img1, &img1_keypoints, &img2, &img2_keypoints, &matches, "./out.tif", );

        assert!(true);
    }

    #[test]
    fn keypoints_count() {
        let img1_dir = "./30.tif";
        let img2_dir = "./31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir);
        let img2: Mat = get_mat_from_dir(img2_dir);

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1);
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2);
        
        println!("{} - Keypoints: {}", img1_dir, img1_keypoints.len());
        println!("{} - Keypoints: {}", img2_dir, img2_keypoints.len());

        assert!(img1_keypoints.len() == 9079 && img2_keypoints.len() == 9357);
    }

    #[test]
    fn knn_matches_count() {
        let img1_dir = "./30.tif";
        let img2_dir = "./31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir);
        let img2: Mat = get_mat_from_dir(img2_dir);

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1);
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2);

        let matches = get_knn_matches(img1_desc, img2_desc, 2, 0.3);

        assert!(matches.len() == 27);
    }

    #[test]
    fn bf_matches_count() {
        let img1_dir = "./30.tif";
        let img2_dir = "./31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir);
        let img2: Mat = get_mat_from_dir(img2_dir);

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1);
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2);

        let matches = get_bruteforce_matches(img1_desc, img2_desc);
        println!("{}", matches.len());

        assert!(matches.len() == 3228);
    }
}
