use cv::{
    core::{perspective_transform, DMatch, KeyPoint, Mat, Point2f, Point2i, Vector, NORM_HAMMING},
    features2d::{AKAZE_DescriptorType, BFMatcher, DrawMatchesFlags, KAZE_DiffusivityType, AKAZE},
    imgcodecs,
    types::{VectorOfDMatch, VectorOfPoint2f, VectorOfVectorOfDMatch},
    Error,
    imgproc::{line, line_def, warp_perspective, warp_perspective_def}
};
use homographier::homographier::Cmat;
use opencv::core::Ptr;

use opencv::{self as cv, prelude::*};

pub fn akaze_keypoint_descriptor_extraction_def(
    img: &Mat,
) -> Result<(Vector<KeyPoint>, Mat), Error> {
    //let img: Mat = cv::imgcodecs::imread(file_location, cv::imgcodecs::IMREAD_COLOR).unwrap();

    let mut akaze: Ptr<AKAZE> = <AKAZE>::create(
        AKAZE_DescriptorType::DESCRIPTOR_MLDB,
        0,
        3,
        0.001f32,
        4,
        4,
        KAZE_DiffusivityType::DIFF_PM_G2
    )?;

    let mut akaze_keypoints = Vector::default();
    let mut akaze_desc = Mat::default();
    let mut dst_img = Mat::default();
    let mask = Mat::default();

    akaze.detect_and_compute(&img, &mask, &mut akaze_keypoints, &mut akaze_desc, false)?;
    cv::features2d::draw_keypoints(
        &img,
        &akaze_keypoints,
        &mut dst_img,
        cv::core::VecN([0., 255., 0., 255.]),
        DrawMatchesFlags::DEFAULT,
    )?;

    Ok((akaze_keypoints, akaze_desc))
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
        for m in &i {
            for n in &i {
                if m.distance < filter_strength * n.distance {
                    good_matches.push(m);
                    break;
                }
            }
        }
    }

    Ok(good_matches)
}

pub fn get_bruteforce_matches(origin_desc: &Mat, target_desc: &Mat) -> Result<Vector<DMatch>, Error> {
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
) -> Result<Mat, Error> {
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

    Ok(out_img)
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

pub fn draw_homography(
    out_img: &mut Mat,
    img1: &Mat,
    homography: &Cmat<f64>
) -> Result<(), Error> {

    let mut object_corners = VectorOfPoint2f::new();
        object_corners.push(Point2f::new(0f32,0f32));
        object_corners.push(Point2f::new(img1.cols() as f32,0f32));
        object_corners.push(Point2f::new(img1.cols() as f32,img1.rows() as f32));
        object_corners.push(Point2f::new(0f32,img1.rows() as f32));

        let mut scene_corners = VectorOfPoint2f::new();

        let _ = perspective_transform(&object_corners, &mut scene_corners, homography);

    for i in 0..4 {
        let _ = line_def(
            out_img,
            Point2i::new(scene_corners.get(i)?.x as i32 + img1.cols(), scene_corners.get(i)?.y as i32 + 0),
            Point2i::new(scene_corners.get((i+1)%4)?.x as i32 + img1.cols(), scene_corners.get((i+1)%4)?.y as i32 + 0),
            opencv::core::VecN::all(255.0)
        );
    }


    Ok(())
}



#[allow(clippy::unwrap_used)]
#[allow(unused_variables)]
#[allow(unused_imports)]
#[allow(dead_code)]
mod test {

    use cv::{core::{perspective_transform, Point2f, Point2i}, imgproc::{line, line_def, warp_perspective, warp_perspective_def}, types::{VectorOfPoint2f, VectorOfVec2f}};
    use opencv::{self as cv, prelude::*};
    use homographier::homographier::*;
    use opencv::imgcodecs;

    use crate::{
        akaze_keypoint_descriptor_extraction_def, draw_homography, export_matches, get_bruteforce_matches, get_knn_matches, get_mat_from_dir, get_points_from_matches
    };

    #[test]
    fn fake_test() {
        let img1_dir = "../resources/test/images/9-2.png";
        let img2_dir = "../resources/test/images/10.png";

        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1).unwrap();
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2).unwrap();

        println!("{} - Keypoints: {}", img1_dir, img1_keypoints.len());
        println!("{} - Keypoints: {}", img2_dir, img2_keypoints.len());

        let matches = get_knn_matches(&img1_desc, &img2_desc, 2, 0.5).unwrap();

        println!("Matches: {}", matches.len());

        let mut out_img = export_matches(
            &img1,
            &img1_keypoints,
            &img2,
            &img2_keypoints,
            &matches,
            "../resources/test/Geotiff/out.png",
        ).unwrap();

        let (img1_matched_points, img2_matched_points) =
            get_points_from_matches(&img1_keypoints, &img2_keypoints, &matches).unwrap();

        println!(
            "Points2f: {} - {}",
            img1_matched_points.len(),
            img2_matched_points.len()
        );

        let mut img1_pts = Vec::new();
        let mut img2_pts = Vec::new();

        for dmatch in matches {
            img1_pts.push(img1_keypoints.get(dmatch.query_idx.try_into().unwrap()).unwrap().pt());
            img2_pts.push(img2_keypoints.get(dmatch.train_idx.try_into().unwrap()).unwrap().pt());
        }

        let res = find_homography_mat(
            &img1_pts,
            &img2_pts,
            Some(HomographyMethod::RANSAC), 
            Some(1f64)
        );

        let res = res.inspect_err(|e| {
            dbg!(e);
        });
        assert!(res.is_ok());

        let res = res.unwrap();
        let homography = res.0;
        let mask = res.1;
        let matches_mask = mask.unwrap();
        let mut dst_img = Mat::default();
        //let mut k = &mut Mat::default();

        //let dst_img = warp_perspective_def(&img1, &mut out_img, &homography, img2.size().unwrap());


        let mut object_corners = VectorOfPoint2f::new();
        object_corners.push(Point2f::new(0f32,0f32));
        object_corners.push(Point2f::new(img1.cols() as f32,0f32));
        object_corners.push(Point2f::new(img1.cols() as f32,img1.rows() as f32));
        object_corners.push(Point2f::new(0f32,img1.rows() as f32));

        let mut scene_corners = VectorOfPoint2f::new();

        let _ = perspective_transform(&object_corners, &mut scene_corners, &homography);

        draw_homography(&mut out_img, &img1, &homography);
        


        imgcodecs::imwrite("../resources/test/Geotiff/out-homo.png", &out_img, &cv::core::Vector::default()).unwrap();
        // Assert for identity matrix
        for col in 0..3 {
            for row in 0..3 {
                if col == row {
                    assert_eq!(&homography.at_2d(row, col).unwrap().round(), &1f64);
                } else {
                    assert_eq!(&homography.at_2d(row, col).unwrap().round(), &0f64);
                }
            }
        }

        assert!(true);
    }

    #[test]
    fn keypoints_count() {
        let img1_dir = "../resources/test/Geotiff/30.tif";
        let img2_dir = "../resources/test/Geotiff/31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1).unwrap();
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2).unwrap();

        println!("{} - Keypoints: {}", img1_dir, img1_keypoints.len());
        println!("{} - Keypoints: {}", img2_dir, img2_keypoints.len());

        assert!(img1_keypoints.len() == 9079 && img2_keypoints.len() == 9357);
    }

    #[test]
    fn knn_matches_count() {
        let img1_dir = "../resources/test/Geotiff/30.tif";
        let img2_dir = "../resources/test/Geotiff/31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1).unwrap();
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2).unwrap();

        let matches = get_knn_matches(&img1_desc, &img2_desc, 2, 0.3).unwrap();

        assert!(matches.len() == 27);
    }

    #[test]
    fn bf_matches_count() {
        let img1_dir = "../resources/test/Geotiff/30.tif";
        let img2_dir = "../resources/test/Geotiff/31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1).unwrap();
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2).unwrap();

        let matches = get_bruteforce_matches(&img1_desc, &img2_desc).unwrap();
        println!("{}", matches.len());

        assert!(matches.len() == 3228);
    }

    // TODO: get_points_from_matches test
}
