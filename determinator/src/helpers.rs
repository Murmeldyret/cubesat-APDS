use std::{
    clone,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use diesel::PgConnection;
use feature_extraction::akaze_keypoint_descriptor_extraction_def;
use homographier::homographier::Cmat;
use opencv::core::{DataType, KeyPoint, MatTraitConst, Vector};
use rgb::alt::BGRA8;

use crate::CameraIntrinsic;

/// where the program should look for reference image(s) and keypoints
pub enum ImgKeypointStore {
    /// Local.0 images and Local.1 keypoints
    Local((PathBuf, PathBuf)),
    Pg(Arc<Mutex<PgConnection>>),
}

impl ImgKeypointStore {
    // TODO: methods for selecting images and/or keypoints within a bounding box
}

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
        panic!("Provided image path does not point to afile")
    }
    match im_path
        .extension()
        .expect("Provided image path has no file extension")
        .to_str()
        .expect("File extention is not valid unicode")
    {
        "png" | "jpg" | "jpeg" | "tiff" | "tif" => {}
        _ => {
            panic!("Provided image uses unsupported file type")
        }
    }
    let path = im_path
        .to_str()
        .expect("provided image path is not valid unicode");

    let image = Cmat::<BGRA8>::imread_checked(&path, -1).expect("Failed to read image ");

    // dbg!(&image);
    let (keypoints, descriptors) = akaze_keypoint_descriptor_extraction_def(&image.mat)
        .expect("AKAZE keypoint extraction failed");

    assert_eq!(
        descriptors.typ(),
        u8::opencv_type(),
        "keypoint descriptors are not of type u8"
    );

    (image, keypoints, Cmat::<u8>::new(descriptors).expect("msg"))
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
