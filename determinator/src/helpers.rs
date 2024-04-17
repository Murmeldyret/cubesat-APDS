use std::{
    clone,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use diesel::PgConnection;
use homographier::homographier::Cmat;
use opencv::core::Mat;

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
            Cmat::from_2d_slice(&arr).map_err(|_err|())?
        }
    };
    Ok(mat)
}

fn parse_into_matrix(path: String) ->Result<Cmat<f64>,()> {
    let path = PathBuf::from(path);
    match path.extension().ok_or(())?.to_str().ok_or(())? {
        "json" => {todo!("har ikke opsat parser endnu :)")}
        _ => Err(()),
        
    }

}