use core::f64;
use std::{
    clone,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use diesel::PgConnection;
use feature_extraction::akaze_keypoint_descriptor_extraction_def;
use homographier::homographier::Cmat;
use opencv::core::{
    DataType, KeyPoint, MatTraitConst, Point2d, Point2f, Point3d, Point3f, Point_, Vector,
};
use rgb::alt::BGRA8;

use crate::CameraIntrinsic;

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

pub fn read_and_extract_kp(im_path: PathBuf) -> (Cmat<BGRA8>, Vector<KeyPoint>, Cmat<u8>) {
    if !im_path.is_file() {
        panic!("{} Provided image path does not point to a file", im_path.to_str().expect("TODO"));
    }

    assert!(matches!(im_path.extension().expect("").to_str(), Some(val) if ["png", "jpg", "jpeg", "tif", "tiff"].contains(&val.to_lowercase().as_str())),"input image file format is unsupported");
    // match im_path
    //     .extension()
    //     .expect("Provided image path has no file extension")
    //     .to_str()
    //     .expect("File extention is not valid unicode")
    // {
    //     "png" | "jpg" | "jpeg" | "tiff" | "tif" => {}
    //     _ => {
    //         panic!("Provided image uses unsupported file type")
    //     }
    // }
    let path = im_path
        .to_str()
        .expect("provided image path is not valid unicode");

    let image = Cmat::<BGRA8>::imread_checked(path, -1).expect("Failed to read image ");

    // dbg!(&image);
    let extracted = akaze_keypoint_descriptor_extraction_def(&image.mat)
        .expect("AKAZE keypoint extraction failed");

    // assert_eq!(
    //     descriptors.typ(),
    //     u8::opencv_type(),
    //     "keypoint descriptors are not of type u8"
    // );

    (image, extracted.keypoints, Cmat::<u8>::new(extracted.descriptors).expect("msg"))
}

pub fn get_camera_matrix(cam_intrins: CameraIntrinsic) -> Result<Cmat<f64>, ()> {
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
