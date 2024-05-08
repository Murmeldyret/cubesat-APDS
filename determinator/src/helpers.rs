use core::f64;
use std::{
    env,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use diesel::Connection;
use feature_database::keypointdb::KeypointDatabase;
use feature_extraction::{
    akaze_keypoint_descriptor_extraction_def, get_knn_matches, get_points_from_matches,
};
use homographier::homographier::{Cmat, ImgObjCorrespondence, PNPRANSACSolution};
use opencv::{
    core::{
        hconcat2, KeyPoint, KeyPointTraitConst, Mat, MatExprTraitConst, MatTraitConst, MatTraitConstManual, Point2d, Point2f, Point3d, Point3f, Point_, Size2i, Vec4d, Vector
    },
    imgcodecs::{IMREAD_COLOR, IMREAD_GRAYSCALE},
};
use rgb::alt::BGR8;

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
        panic!("{:?} Provided image path does not point to a file", im_path);
    }

    assert!(
        matches!(im_path.extension().expect("Image has no file extension").to_str(), Some(val) if ["png", "jpg", "jpeg", "tif", "tiff"].contains(&val.to_lowercase().as_str())),
        "input image of file format {:?} is unsupported",
        im_path.extension()
    );
    let path = im_path
        .to_str()
        .expect("provided image path is not valid unicode");

    // it is assumed that input images will not contain an alpha channel
    let image = Cmat::<BGR8>::imread_checked(path, IMREAD_COLOR).expect("Failed to read image");

    let extracted = akaze_keypoint_descriptor_extraction_def(&image.mat)
        .expect("AKAZE keypoint extraction failed");

    ReadAndExtractKpResult(
        image,
        extracted.keypoints,
        Cmat::<u8>::new(extracted.descriptors).expect("Matrix construction should not fail"),
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
            unimplemented!("har ikke opsat parser endnu :)")
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
        .expect("keypoint matching failed"),
        None => {
            let mut img_points: Vector<Point2f> = Vector::new();
            opencv::calib3d::find_chessboard_corners_def(
                &Cmat::<u8>::imread_checked(&args.img_path.to_string_lossy(), IMREAD_GRAYSCALE)
                    .expect("failed to read image")
                    .mat,
                Size2i::new(7, 7),
                &mut img_points,
            )
            .expect("failed to find chessboard corners")
            .then_some(())
            .expect("failed to find chessboard corners");
            assert_eq!(ref_keypoints.len(), img_points.len());
            let corres = ref_keypoints
                .into_iter()
                .zip(img_points)
                .map(|(o, i)| {
                    ImgObjCorrespondence::new(
                        Point3d::from_point(o.pt().to().expect("f32 cast to f64 should not fail")),
                        i.to().expect("f32 cast to f64 should not fail"),
                    )
                })
                .collect::<Vec<_>>();
            corres
        }
    }
}

fn matching_with_descriptors(
    img: &ReadAndExtractKpResult,
    ref_desc: &Cmat<u8>,
    ref_kp: Vector<KeyPoint>,
) -> Result<Vec<ImgObjCorrespondence>, opencv::Error> {
    let matches = get_knn_matches(&img.2.mat, &ref_desc.mat, 2, 0.8)?;

    let (img_points, obj_points_2d) = get_points_from_matches(&img.1, &ref_kp, &matches)?;
    assert_eq!(img_points.len(), obj_points_2d.len());

    // map object points to real world coordinates
    let obj_points = get_3d_world_coord_from_2d_point(
        obj_points_2d
            .into_iter()
            .map(|f| Point2d::new(f.x as f64, f.y as f64))
            .collect(),
        todo!("det er ikke helt klart endnu"),
    );

    Ok(point_pair_to_correspondence(
        img_points,
        obj_points
            .into_iter()
            .map(|f| Point3f::new(f.x as f32, f.y as f32, f.z as f32))
            .collect::<Vec<_>>(),
    ))
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
            let points: Result<Vec<KeyPoint>, _> = (1..=7)
                .map(|f| (f, (1..=7)))
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

// TODO: kan godt være topo parameter skal ændres til en anden type
pub fn get_3d_world_coord_from_2d_point(points: Vec<Point2d>, db: DbType) -> Vec<Point3d> {
    points
        .into_iter()
        .map(|p| {
            let world_coord: (f64, f64, f64) = todo!("@rasmus"); //@Murmeldyret
            Point3d::new(world_coord.0, world_coord.1, world_coord.2)
        })
        .collect::<Vec<_>>()
}

pub fn point2f_to2d(p: Point2f) -> Point2d {
    Point2d::new(p.x as f64, p.y as f64)
}

pub fn point3f_to3d(p: Point3f) -> Point3d {
    Point3d::new(p.x as f64, p.y as f64, p.z as f64)
}
// https://docs.opencv.org/4.x/dc/d2c/tutorial_real_time_pose.html
/// # Returns
/// a 2d homogenous point (z=1)
pub fn project_obj_point(
    obj_point: Point3d,
    solution: PNPRANSACSolution,
    cam_mat: Cmat<f64>,
) -> Point3d {
    // let tvec = [first,second,third];
    let mut rt_mat = Cmat::<f64>::zeros(4, 4).expect("Matrix initialization should not fail");
    hconcat2(&solution.rvec.mat, &solution.tvec.mat, &mut rt_mat.mat)
        .expect("Matrix operation should not fail");
    let rt_mat = rt_mat;

    // homogenous object point
    let obj_point_hom = Vec4d::new(obj_point.x, obj_point.y, obj_point.z, 1f64).to_vec();

    // dbg!(&obj_point_hom);
    // println!("{:?}\n{:?}\n{:?}\n",rt_mat.mat.at_row::<f64>(0).unwrap(),rt_mat.mat.at_row::<f64>(1).unwrap(),rt_mat.mat.at_row::<f64>(2).unwrap());
    let obj_point_hom_mat =
        Mat::from_slice(&Vec4d::new(obj_point.x, obj_point.y, obj_point.z, 1f64).to_vec())
            .expect("Matrix construction should not fail");

    let mut temp = (cam_mat.mat * rt_mat.mat)
        .into_result()
        .expect("matrix expression should not failed")
        .to_mat()
        .expect("matrix construction should not fail");

    assert_eq!(temp.rows(),3,"Matrix multiplication between calibration matrix (A) and rotation+translation matix (Rt) should yield a 3X4 matrix");
    assert_eq!(temp.cols(),4,"Matrix multiplication between calibration matrix (A) and rotation+translation matix (Rt) should yield a 3X4 matrix");

    // println!("{:?}\n{:?}\n{:?}\n",temp.at_row::<f64>(0).unwrap(),temp.at_row::<f64>(1).unwrap(),temp.at_row::<f64>(2).unwrap());
    let mut result: [f64; 3] = [0f64; 3];
    // this is because opencv does not allow matrix vector product for some reason :(
    for (i, e) in result.iter_mut().enumerate().take(temp.rows() as usize) {
        *e = temp
            .at_row::<f64>(i as i32)
            .expect("row index should be within range")
            .iter()
            .enumerate()
            .map(|(i, e)| {
                e * obj_point_hom
                    .get(i)
                    .expect("index should not be out of range")
            })
            // .inspect(|f|{dbg!(f);})
            .reduce(|acc, elem| acc + elem)
            .expect("Reduce operation should yield a value");
    }
    Point3d::new(result[0] / result[2], result[1] / result[2], 1f64)
    // let vector = temp.iter::<f64>().unwrap().map(|(p,e)|e*obj_point_hom.get(p.y as usize).unwrap()).collect::<Vec<f64>>();
    // let rhs = dbg!(temp.dot(&obj_point_hom));
    // dbg!(&rhs);
    // let _ = temp.iter::<f64>().unwrap().inspect(|f|println!("({},{}) = {}",f.0.x,f.0.y,f.1)).collect::<Vec<_>>();
    // let rhs = temp.elem_mul(obj_point_hom).into_result().unwrap().to_mat().unwrap();
}
