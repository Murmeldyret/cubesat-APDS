use std::{
    clone,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use diesel::PgConnection;

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
