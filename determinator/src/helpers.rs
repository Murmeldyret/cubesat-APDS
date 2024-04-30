use core::f64;
use std::{
    clone, env,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use diesel::{Connection, PgConnection};
use feature_database::keypointdb::KeypointDatabase;
use feature_extraction::{
    akaze_keypoint_descriptor_extraction_def, get_knn_matches, get_points_from_matches,
};
use homographier::homographier::{Cmat, ImgObjCorrespondence};
use opencv::{
    core::{
        DataType, KeyPoint, MatTraitConst, Point2d, Point2f, Point3d, Point3f, Point_, Size2i,
        Size_, Vector,
    },
    imgcodecs::{IMREAD_COLOR, IMREAD_GRAYSCALE},
};
use rgb::alt::{BGR8, BGRA8};

use crate::{Args, CameraIntrinsic, DbType};

#[derive(Debug, Clone)]
pub enum Coordinates3d {
    Ellipsoidal { lat: f32, lon: f32, height: f32 },
    Cartesian { x: f32, y: f32, z: f32 },
}

#[derive(Debug, Clone)]
pub enum Coordinates2d {
    Ellipsoidal { lat: f32, lon: f32 },
    Cartesian { x: f32, y: f32 },
}

impl From<Coordinates3d> for Coordinates2d {
    fn from(value: Coordinates3d) -> Self {
        match value {
            Coordinates3d::Ellipsoidal {
                lat,
                lon,
                height: _,
            } => Coordinates2d::Ellipsoidal { lat, lon },
            Coordinates3d::Cartesian { x, y, z: _ } => Coordinates2d::Cartesian { x, y },
        }
    }
}

pub struct ReadAndExtractKpResult(pub Cmat<BGR8>, pub Vector<KeyPoint>, pub Cmat<u8>);

pub fn read_and_extract_kp(im_path: &Path) -> ReadAndExtractKpResult {
    if !im_path.is_file() {
        panic!(
            "{} Provided image path does not point to a file",
            im_path.to_str().expect("TODO")
        );
    }

    assert!(
        matches!(im_path.extension().expect("").to_str(), Some(val) if ["png", "jpg", "jpeg", "tif", "tiff"].contains(&val.to_lowercase().as_str())),
        "input image file format is unsupported"
    );
    let path = im_path
        .to_str()
        .expect("provided image path is not valid unicode");

    // it is assumed that input images will not contain an alpha channel
    let image = Cmat::<BGR8>::imread_checked(path, IMREAD_COLOR).expect("Failed to read image");

    // dbg!(&image);
    let extracted = akaze_keypoint_descriptor_extraction_def(&image.mat)
        .expect("AKAZE keypoint extraction failed");

    // assert_eq!(
    //     descriptors.typ(),
    //     u8::opencv_type(),
    //     "keypoint descriptors are not of type u8"
    // );

    ReadAndExtractKpResult(
        image,
        extracted.keypoints,
        Cmat::<u8>::new(extracted.descriptors).expect("msg"),
    )
}

pub fn get_camera_matrix(cam_intrins: CameraIntrinsic) -> Result<Cmat<f64>, ()> {
    // [[f_x,s  ,c_x],
    //  [0.0,f_y,c_y],
    //  [0.0,0.0,1.0]]
    let mat: Cmat<f64> = match cam_intrins {
        CameraIntrinsic::File { path } => parse_into_matrix(path)?,
        CameraIntrinsic::Manual {
            focal_len_x,
            focal_len_y,
            skew,
            offset_x,
            offset_y,
        } => {
            let arr: [[f64; 3]; 3] = [
                [focal_len_x, skew, offset_x],
                [0.0, focal_len_y, offset_y],
                [0.0, 0.0, 1.0],
            ];
            Cmat::from_2d_slice(&arr).map_err(|_err| ())?
        }
    };
    Ok(mat)
}

fn parse_into_matrix(path: String) -> Result<Cmat<f64>, ()> {
    let path = PathBuf::from(path);
    match path.extension().ok_or(())?.to_str().ok_or(())? {
        "json" => {
            todo!("har ikke opsat parser endnu :)")
        }
        _ => Err(()),
    }
}

pub fn img_obj_corres(args: &Args, image: ReadAndExtractKpResult) -> Vec<ImgObjCorrespondence> {
    let (ref_keypoints, ref_descriptors) = ref_keypoints(args);

    match ref_descriptors {
        Some(val) => matching_with_descriptors(
            &image,
            &Cmat::<u8>::from_2d_slice(&val).expect("Failed to convert keypoints to matrix"),
            Vector::from_iter(ref_keypoints),
        )
        .expect("TODO"),
        None => {
            // let points = get_points_from_matches(image.1, ref_keypoints, matches)
            let mut img_points: Vector<Point2f> = Vector::new();
            let _ = opencv::calib3d::find_chessboard_corners_def(
                &Cmat::<u8>::imread_checked(&args.img_path.to_string_lossy(), IMREAD_GRAYSCALE)
                    .expect("failed to read image")
                    .mat,
                Size2i::new(7, 7),
                &mut img_points,
            ).expect("msg").then_some(()).expect("msg");
            // let obj_points = point2d_to_3d(ref_keypoints, todo!());
            // point_pair_to_correspondence(image.1, obj_points)
            todo!()
        }
    }
}

fn matching_with_descriptors(
    img: &ReadAndExtractKpResult,
    ref_desc: &Cmat<u8>,
    ref_kp: Vector<KeyPoint>,
) -> Result<Vec<ImgObjCorrespondence>, opencv::Error> {
    let matches = get_knn_matches(&img.0.mat, &ref_desc.mat, 2, 0.8)?;

    let (img_points, obj_points_2d) = get_points_from_matches(&img.1, &ref_kp, &matches)?;
    assert_eq!(img_points.len(), obj_points_2d.len());

    let obj_points = point2d_to_3d(obj_points_2d.into_iter().collect(), todo!());

    Ok(point_pair_to_correspondence(img_points, obj_points))
}

fn point_pair_to_correspondence(
    img_points: Vector<Point_<f32>>,
    obj_points: Vec<opencv::core::Point3_<f32>>,
) -> Vec<ImgObjCorrespondence> {
    img_points
        .into_iter()
        .zip(obj_points)
        .map(|f| ImgObjCorrespondence::new(point3f_to3d(f.1), point2f_to2d(f.0)))
        .collect()
}

pub fn ref_keypoints(args: &Args) -> (Vec<KeyPoint>, Option<Vec<Vec<u8>>>) {
    match args.demo {
        true => {
            // TIHI @Murmeldyret, here be no side effects
            let points: Result<Vec<KeyPoint>, _> = (1..7)
                .map(|f| (f, (1..7)))
                .flat_map(|row| {
                    row.1
                        .map(move |col| KeyPoint::new_coords_def(row.0 as f32, col as f32, 1.0))
                })
                .collect();
            (points.expect("Failed to create keypoints"), None)
        }
        false => {
            let (keypoints_from_db, descriptors) = keypoints_from_db(
                &env::var("DATABASE_URL").expect("failed to read DATABASE_URL"),
                args,
            );
            (keypoints_from_db, Some(descriptors))
        }
    }
}

fn keypoints_from_db(conn_url: &str, arg: &Args) -> (Vec<KeyPoint>, Vec<Vec<u8>>) {
    let conn: DbType = Arc::new(Mutex::new(
        Connection::establish(conn_url).expect("Failed to connect to database"),
    ));

    // retrieve keypoints from mosaic image
    let ref_keypoints = feature_database::keypointdb::Keypoint::read_keypoints_from_lod(
        &mut conn.lock().expect("Mutex poisoning"),
        arg.lod,
    )
    .expect("Failed to query database");
    // Map keypoints to opencv compatible type
    let (ref_keypoints, ref_descriptors) = db_kp_to_opencv_kp(ref_keypoints);

    (ref_keypoints, ref_descriptors)
}

fn db_kp_to_opencv_kp(
    ref_keypoints: Vec<feature_database::models::Keypoint>,
) -> (Vec<KeyPoint>, Vec<Vec<u8>>) {
    let (ref_keypoints, ref_descriptors): (Vec<_>, Vec<_>) = ref_keypoints
        .into_iter()
        .map(|f| {
            (
                opencv::core::KeyPoint::new_point(
                    Point2f::new(f.x_coord, f.y_coord),
                    f.size,
                    f.angle,
                    f.response,
                    f.octave,
                    f.class_id,
                )
                .expect("error in converting db keypoint to opencv keypoint"),
                f.descriptor,
            )
        })
        .unzip();
    (ref_keypoints, ref_descriptors)
}

///
pub fn point2d_to_3d(points: Vec<Point2f>, topo: Cmat<f32>) -> Vec<Point3f> {
    points
        .into_iter()
        .map(|p| {
            Point3f::new(
                p.x,
                p.y,
                *topo
                    .at_2d(p.x.floor() as i32, p.y.floor() as i32)
                    .expect("Out of bounds"),
            )
        })
        .collect::<Vec<_>>()
}

pub fn point2f_to2d(p: Point2f) -> Point2d {
    Point2d::new(p.x as f64, p.y as f64)
}

pub fn point3f_to3d(p: Point3f) -> Point3d {
    Point3d::new(p.x as f64, p.y as f64, p.z as f64)
}
