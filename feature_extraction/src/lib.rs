use cv::{
    core::{DMatch, KeyPoint, Mat, Point2f, Vector, NORM_HAMMING},
    features2d::{AKAZE_DescriptorType, BFMatcher, DrawMatchesFlags, KAZE_DiffusivityType, AKAZE},
    imgcodecs,
    types::{VectorOfDMatch, VectorOfPoint2f, VectorOfVectorOfDMatch},
    Error,
};
use opencv::core::Ptr;

use opencv::{self as cv, prelude::*};

pub struct ExtractedKeyPoint {
    keypoints: Vector<KeyPoint>,
    descriptors: Mat,
}

#[derive(Debug)]
pub struct DbKeypoints {
    pub x_coord: f32,
    pub y_coord: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
    pub class_id: i32,
    pub descriptor: Vec<u8>,
    pub image_id: i32,
}

impl ExtractedKeyPoint {
    pub fn to_db_type(&self, image_id: i32) -> Vec<DbKeypoints> {
        let keypoints = self.keypoints.to_vec();

        let mut db_keypoints: Vec<DbKeypoints> = Vec::with_capacity(self.keypoints.len());

        for (i, keypoint) in keypoints.iter().enumerate() {
            db_keypoints.push(DbKeypoints {
                x_coord: keypoint.pt().x,
                y_coord: keypoint.pt().y,
                size: keypoint.size(),
                angle: keypoint.angle(),
                response: keypoint.response(),
                octave: keypoint.octave(),
                class_id: keypoint.class_id(),
                descriptor: self
                    .descriptors
                    .at_row(i.try_into().expect("Could not cast i"))
                    .expect("Could not find descriptor")
                    .to_vec(),
                image_id,
            });
        }

        db_keypoints
    }
}

pub fn akaze_keypoint_descriptor_extraction_def(img: &Mat) -> Result<ExtractedKeyPoint, Error> {
    //let img: Mat = cv::imgcodecs::imread(file_location, cv::imgcodecs::IMREAD_COLOR).unwrap();

    let mut akaze: Ptr<AKAZE> = <AKAZE>::create(
        AKAZE_DescriptorType::DESCRIPTOR_MLDB,
        0,
        3,
        0.001f32,
        4,
        4,
        KAZE_DiffusivityType::DIFF_PM_G2,
        -1,
    )?;

    let mut akaze_keypoints = Vector::default();
    let mut akaze_desc: Mat = Mat::default();
    let mask = Mat::default();

    akaze.detect_and_compute(&img, &mask, &mut akaze_keypoints, &mut akaze_desc, false)?;
    // cv::features2d::draw_keypoints(
    //     &img,
    //     &akaze_keypoints,
    //     &mut dst_img,
    //     cv::core::VecN([0., 255., 0., 255.]),
    //     DrawMatchesFlags::DEFAULT,
    // )?;

    Ok(ExtractedKeyPoint {
        keypoints: akaze_keypoints,
        descriptors: akaze_desc,
    })
}

pub fn get_knn_matches(
    origin_desc: &Mat,
    target_desc: &Mat,
    k: i32,
    filter_strength: f32,
) -> Result<Vector<DMatch>, Error> {
    let mut matches = VectorOfVectorOfDMatch::new();
    let bf_matcher = BFMatcher::new(NORM_HAMMING, false)?;

    bf_matcher.knn_train_match_def(&origin_desc, &target_desc, &mut matches, k)?;

    let mut good_matches = VectorOfDMatch::new();

    for i in &matches {
        if i.get(0)?.distance < i.get(1)?.distance * filter_strength {
            good_matches.push(i.get(0)?);
        } 
    }

    Ok(good_matches)
}

pub fn get_bruteforce_matches(
    origin_desc: &Mat,
    target_desc: &Mat,
) -> Result<Vector<DMatch>, Error> {
    let mut matches = VectorOfDMatch::new();
    let bf_matcher = BFMatcher::new(NORM_HAMMING, true)?;

    bf_matcher.train_match_def(&origin_desc, &target_desc, &mut matches)?;

    Ok(matches)
}

pub fn export_matches(
    img1: &Mat,
    img1_keypoints: &Vector<KeyPoint>,
    img2: &Mat,
    img2_keypoints: &Vector<KeyPoint>,
    matches: &Vector<DMatch>,
    export_location: &str,
) -> Result<(), Error> {
    let mut out_img = Mat::default();
    let matches_mask = Vector::new();

    cv::features2d::draw_matches(
        &img1,
        img1_keypoints,
        &img2,
        img2_keypoints,
        matches,
        &mut out_img,
        opencv::core::VecN::all(-1.0),
        opencv::core::VecN::all(-1.0),
        &matches_mask,
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS,
    )?;

    imgcodecs::imwrite(export_location, &out_img, &Vector::default())?;

    Ok(())
}

pub fn get_mat_from_dir(img_dir: &str) -> Result<Mat, Error> {
    imgcodecs::imread(img_dir, imgcodecs::IMREAD_COLOR)
}

pub fn get_points_from_matches(
    img1_keypoints: &Vector<KeyPoint>,
    img2_keypoints: &Vector<KeyPoint>,
    matches: &Vector<DMatch>,
) -> Result<(Vector<Point2f>, Vector<Point2f>), Error> {
    let mut img1_matched_keypoints: Vector<KeyPoint> = Vector::default();
    let mut img2_matched_keypoints: Vector<KeyPoint> = Vector::default();
    for m in matches {
        img1_matched_keypoints.push(img1_keypoints.get(m.img_idx.try_into()?)?);
        img2_matched_keypoints.push(img2_keypoints.get(m.train_idx.try_into()?)?);
    }

    let mut img1_matched_points: Vector<Point2f> = VectorOfPoint2f::new();
    let mut img2_matched_points: Vector<Point2f> = VectorOfPoint2f::new();

    opencv::core::KeyPoint::convert_def(&img1_matched_keypoints, &mut img1_matched_points)?;
    opencv::core::KeyPoint::convert_def(&img1_matched_keypoints, &mut img2_matched_points)?;

    Ok((img1_matched_points, img2_matched_points))
}

#[allow(clippy::unwrap_used)]
#[allow(unused_variables)]
#[allow(unused_imports)]
#[allow(dead_code)]
mod test {

    use cv::core::Point2f;
    use opencv::{self as cv, prelude::*};

    use crate::{
        akaze_keypoint_descriptor_extraction_def, export_matches, get_bruteforce_matches,
        get_knn_matches, get_mat_from_dir, get_points_from_matches,
    };

    #[test]
    fn fake_test() {
        let img1_dir = "../resources/test/Geotiff/30.tif";
        let img2_dir = "../resources/test/Geotiff/31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        let img1_keypoints = akaze_keypoint_descriptor_extraction_def(&img1).unwrap();
        let img2_keypoints = akaze_keypoint_descriptor_extraction_def(&img2).unwrap();

        println!(
            "{} - Keypoints: {}",
            img1_dir,
            img1_keypoints.keypoints.len()
        );
        println!(
            "{} - Keypoints: {}",
            img2_dir,
            img2_keypoints.keypoints.len()
        );

        let matches = get_knn_matches(
            &img1_keypoints.descriptors,
            &img2_keypoints.descriptors,
            2,
            0.3,
        )
        .unwrap();

        println!("Matches: {}", matches.len());

        let _ = export_matches(
            &img1,
            &img1_keypoints.keypoints,
            &img2,
            &img2_keypoints.keypoints,
            &matches,
            "../resources/test/Geotiff/out.tif",
        );

        let (img1_matched_points, img2_matched_points) = get_points_from_matches(
            &img1_keypoints.keypoints,
            &img2_keypoints.keypoints,
            &matches,
        )
        .unwrap();

        println!(
            "Points2f: {} - {}",
            img1_matched_points.len(),
            img2_matched_points.len()
        );

        assert!(true);
    }

    #[test]
    fn keypoints_count() {
        let img1_dir = "../resources/test/Geotiff/30.tif";
        let img2_dir = "../resources/test/Geotiff/31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        let img1_keypoints = akaze_keypoint_descriptor_extraction_def(&img1).unwrap().keypoints;
        let img2_keypoints = akaze_keypoint_descriptor_extraction_def(&img2).unwrap().keypoints;

        println!(
            "{} - Keypoints: {}",
            img1_dir,
            img1_keypoints.keypoints.len()
        );
        println!(
            "{} - Keypoints: {}",
            img2_dir,
            img2_keypoints.keypoints.len()
        );

        assert!(img1_keypoints.len() == 9079 && img2_keypoints.len() == 9357);
    }

    #[test]
    fn knn_matches_count() {
        let img1_dir = "../resources/test/Geotiff/30.tif";
        let img2_dir = "../resources/test/Geotiff/31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        let img1_keypoints = akaze_keypoint_descriptor_extraction_def(&img1).unwrap();
        let img2_keypoints = akaze_keypoint_descriptor_extraction_def(&img2).unwrap();

        let matches = get_knn_matches(
            &img1_keypoints.descriptors,
            &img2_keypoints.descriptors,
            2,
            0.3,
        )
        .unwrap();

        assert!(matches.len() == 27);
    }

    #[test]
    fn bf_matches_count() {
        let img1_dir = "../resources/test/Geotiff/30.tif";
        let img2_dir = "../resources/test/Geotiff/31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        let img1_keypoints = akaze_keypoint_descriptor_extraction_def(&img1).unwrap();
        let img2_keypoints = akaze_keypoint_descriptor_extraction_def(&img2).unwrap();

        let matches =
            get_bruteforce_matches(&img1_keypoints.descriptors, &img2_keypoints.descriptors)
                .unwrap();
        println!("{}", matches.len());

        assert!(matches.len() == 3228);
    }

    // TODO: get_points_from_matches test
}
