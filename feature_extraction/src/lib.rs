use cv::{
    core::{perspective_transform, DMatch, KeyPoint, Mat, Point2d, Point2f, Point2i, Vector, NORM_HAMMING},
    features2d::{AKAZE_DescriptorType, BFMatcher, DrawMatchesFlags, KAZE_DiffusivityType, AKAZE},
    imgcodecs,
    imgproc::{line, LINE_8},
    types::{VectorOfDMatch, VectorOfPoint2d, VectorOfPoint2f, VectorOfVectorOfDMatch},
    Error,
};
use homographier::homographier::Cmat;
use opencv::core::Ptr;

use opencv::{self as cv, prelude::*};

pub fn akaze_keypoint_descriptor_extraction_def(
    img: &Mat,
    max_points: Option<i32>
) -> Result<(Vector<KeyPoint>, Mat), Error> {
    let mut akaze: Ptr<AKAZE> = <AKAZE>::create(
        AKAZE_DescriptorType::DESCRIPTOR_MLDB,
        0,
        3,
        0.001f32,
        4,
        4,
        KAZE_DiffusivityType::DIFF_PM_G2,
        max_points.unwrap_or(262143)
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

pub fn get_object_and_scene_corners(
    img1: &Mat,
    homography: &Cmat<f64>,
) -> Result<(Vector<Point2f>, Vector<Point2f>), Error> {
    let mut object_corners = VectorOfPoint2f::new();
    object_corners.push(Point2f::new(0f32, 0f32));
    object_corners.push(Point2f::new(img1.cols() as f32, 0f32));
    object_corners.push(Point2f::new(img1.cols() as f32, img1.rows() as f32));
    object_corners.push(Point2f::new(0f32, img1.rows() as f32));

    let mut scene_corners = VectorOfPoint2f::new();

    let _ = perspective_transform(&object_corners, &mut scene_corners, homography);

    Ok((object_corners, scene_corners))
}

/// Images have to be geotiff since world coordinates are being extracted
pub fn get_lat_long(
    img1: Mat,
    img2_dir: &str,
    homography: &Cmat<f64>
) -> Result<Vector<Point2d>, Error> {

    let (_object_corners, scene_corners) = get_object_and_scene_corners(&img1, homography)?;

    let img2: Mat = get_mat_from_dir(img2_dir)?;
    let dataset = gdal::Dataset::open(img2_dir);
    let projected_dataset = dataset.unwrap().geo_transform().unwrap();
    println!("{:#?}", &projected_dataset);
    let x_min = projected_dataset.get(0).unwrap().abs();
    let x_size = projected_dataset.get(1).unwrap().abs();
    let y_min = projected_dataset.get(3).unwrap().abs();
    let y_size = projected_dataset.get(5).unwrap().abs();

    let ref_img_width = img2.size()?.width;
    let ref_img_height = img2.size()?.height;

    println!("width: {}, height: {}", ref_img_width, ref_img_height);  // TODO: remove this

    let mut projected_geo_coords = VectorOfPoint2d::new();

    for i in 0..4 {
        projected_geo_coords.insert(
            i,
            Point2d::new(
                scene_corners.get(i)?.y as f64 * y_size + y_min,
                scene_corners.get(i)?.x as f64 * x_size + x_min
            )
        )?;

        println!(
            "Lat: {}, Long {}",
            projected_geo_coords.get(i)?.x,
            projected_geo_coords.get(i)?.y
        );  // TODO: remove this

        println!(
            "x: {}, y: {}",
            scene_corners.get(i)?.x,
            scene_corners.get(i)?.y
        ); // TODO: remove this
    }

    Ok(projected_geo_coords)

}

pub fn draw_homography_lines(
    out_img: &mut Mat,
    img1: &Mat,
    homography: &Cmat<f64>,
) -> Result<(), Error> {
    let (_object_corners, scene_corners) = get_object_and_scene_corners(img1, homography)?;

    for i in 0..4 {
        let _ = line(
            out_img,
            Point2i::new(
                scene_corners.get(i)?.x as i32 + img1.cols(),
                scene_corners.get(i)?.y as i32,
            ),
            Point2i::new(
                scene_corners.get((i + 1) % 4)?.x as i32 + img1.cols(),
                scene_corners.get((i + 1) % 4)?.y as i32,
            ),
            opencv::core::VecN::new(0.0, 0.0, 255.0, 0.0),
            2i32,
            LINE_8,
            0,
        );
    }

    Ok(())
}

fn get_matched_points_vec(
    img1_keypoints: &Vector<KeyPoint>,
    img2_keypoints: &Vector<KeyPoint>,
    matches: &Vector<DMatch>,
) -> Result<(Vec<Point2f>, Vec<Point2f>), Error> {
    let mut img1_matched_points = Vec::new();
    let mut img2_matched_points = Vec::new();

    for dmatch in matches {
        img1_matched_points.push(
            img1_keypoints
                .get(dmatch.query_idx.try_into()?)?
                .pt(),
        );
        img2_matched_points.push(
            img2_keypoints
                .get(dmatch.train_idx.try_into()?)?
                .pt(),
        );

    }

    Ok((img1_matched_points, img2_matched_points))
}
#[allow(clippy::unwrap_used)]
#[allow(unused_variables)]
#[allow(unused_imports)]
#[allow(dead_code)]
mod test {

    use cv::{
        core::{perspective_transform, Point2f, Point2i},
        imgproc::{line, line_def, warp_perspective, warp_perspective_def},
        types::{VectorOfPoint2f, VectorOfVec2f},
    };
    use homographier::homographier::*;
    use opencv::imgcodecs;
    use opencv::{self as cv, prelude::*};

    use crate::{
        akaze_keypoint_descriptor_extraction_def, draw_homography_lines, export_matches, get_bruteforce_matches, get_knn_matches, get_lat_long, get_mat_from_dir, get_matched_points_vec, get_points_from_matches
    };

    use gdal::*;

    #[test]
    fn fake_test() {
        // Loads in the two images, img1 = query, img2 = reference
        let img1_dir = "../resources/test/benchmark/Denmark_small.png";
        let img2_dir = "../resources/test/benchmark/Denmark_8192.png";
        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        // Gets keypoints and decsriptors using AKAZE
        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1, Some(25)).unwrap();
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2, None).unwrap();

        println!("{} - Keypoints: {}", img1_dir, img1_keypoints.len());
        println!("{} - Keypoints: {}", img2_dir, img2_keypoints.len());

        // Gets k(2)nn matches using Lowe's distance ratio
        let matches = get_knn_matches(&img1_desc, &img2_desc, 2, 0.7).unwrap();
        println!("Matches: {}", matches.len());

        // Exports an image containing img1 & img2 and draws lines between their matches
        let mut out_img = export_matches(
            &img1,
            &img1_keypoints,
            &img2,
            &img2_keypoints,
            &matches,
            "../resources/test/Geotiff/out.png",
        )
        .unwrap();

        // Find homography wants a Vec<Point2f> instead of a Vector<Point2f>
        let (img1_matched_points_vec, img2_matched_points_vec) =
            get_matched_points_vec(&img1_keypoints, &img2_keypoints, &matches).unwrap();

        let res = find_homography_mat(
            &img1_matched_points_vec,
            &img2_matched_points_vec,
            Some(HomographyMethod::RANSAC),
            Some(10f64),
        );

        let res = res.inspect_err(|e| {
            dbg!(e);
        });
        assert!(res.is_ok());
        
        let res = res.unwrap();
        let homography = res.0;
        let mask = res.1;
        let matches_mask = mask.unwrap();

        // Draws a projection of where the query image is on the reference image
        let _ = draw_homography_lines(&mut out_img, &img1, &homography);

        //let _ = get_lat_long(img1, img2_dir, &homography);

        imgcodecs::imwrite(
            "../resources/test/Geotiff/out-homo.png",
            &out_img,
            &cv::core::Vector::default(),
        )
        .unwrap();

        assert!(true);
    }

    #[test]
    fn keypoints_count() {
        let img1_dir = "../resources/test/Geotiff/30.tif";
        let img2_dir = "../resources/test/Geotiff/31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1, None).unwrap();
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2, None).unwrap();

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

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1, None).unwrap();
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2, None).unwrap();

        let matches = get_knn_matches(&img1_desc, &img2_desc, 2, 0.3).unwrap();

        assert!(matches.len() == 27);
    }

    #[test]
    fn bf_matches_count() {
        let img1_dir = "../resources/test/Geotiff/30.tif";
        let img2_dir = "../resources/test/Geotiff/31.tif";

        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1, None).unwrap();
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2, None).unwrap();

        let matches = get_bruteforce_matches(&img1_desc, &img2_desc).unwrap();
        println!("{}", matches.len());

        assert!(matches.len() == 3228);
    }

    #[test]
    fn coords_test() {
        // Loads in the two images, img1 = query, img2 = reference
        let img1_dir = "../resources/test/Geotiff/31.tif";
        let img2_dir = "../resources/test/Geotiff/30.tif";
        let img1: Mat = get_mat_from_dir(img1_dir).unwrap();
        let img2: Mat = get_mat_from_dir(img2_dir).unwrap();

        // Gets keypoints and decsriptors using AKAZE
        let (img1_keypoints, img1_desc) = akaze_keypoint_descriptor_extraction_def(&img1, None).unwrap();
        let (img2_keypoints, img2_desc) = akaze_keypoint_descriptor_extraction_def(&img2, None).unwrap();

        println!("{} - Keypoints: {}", img1_dir, img1_keypoints.len());
        println!("{} - Keypoints: {}", img2_dir, img2_keypoints.len());

        // Gets k(2)nn matches using Lowe's distance ratio
        let matches = get_knn_matches(&img1_desc, &img2_desc, 2, 0.7).unwrap();
        println!("Matches: {}", matches.len());

        // Exports an image containing img1 & img2 and draws lines between their matches
        let mut out_img = export_matches(
            &img1,
            &img1_keypoints,
            &img2,
            &img2_keypoints,
            &matches,
            "../resources/test/Geotiff/out.png",
        )
        .unwrap();

        // Find homography wants a Vec<Point2f> instead of a Vector<Point2f>
        let (img1_matched_points_vec, img2_matched_points_vec) =
            get_matched_points_vec(&img1_keypoints, &img2_keypoints, &matches).unwrap();

        let res = find_homography_mat(
            &img1_matched_points_vec,
            &img2_matched_points_vec,
            Some(HomographyMethod::RANSAC),
            Some(10f64),
        );

        let res = res.inspect_err(|e| {
            dbg!(e);
        });
        assert!(res.is_ok());
        
        let res = res.unwrap();
        let homography = res.0;

        let coords = get_lat_long(img1, img2_dir, &homography).unwrap();

        imgcodecs::imwrite(
            "../resources/test/Geotiff/out-homo.png",
            &out_img,
            &cv::core::Vector::default(),
        )
        .unwrap();

        assert!(coords.get(0).unwrap().x == 57.58392892466675);
        assert!(coords.get(0).unwrap().y == 9.849494991472618);
        assert!(coords.get(2).unwrap().x == 57.757949696191034);
        assert!(coords.get(2).unwrap().y == 10.100162414469677);
    }

    
}
