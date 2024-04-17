use clap::{Parser, Subcommand};
use dotenvy::dotenv;

use feature_database::models::Keypoint;
use feature_database::schema::keypoint::descriptor;
use feature_extraction::{
    akaze_keypoint_descriptor_extraction_def, get_knn_matches, get_points_from_matches,
};
use helpers::{get_camera_matrix, Coordinates3d};
use homographier::homographier::{raster_to_mat, Cmat};

use diesel::PgConnection;
use opencv::core::{Point2f, Point3f};
use rgb::alt::BGRA;
use rgb::{alt::BGRA8, RGBA};
use std::path::Path;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub mod helpers;

///TODO:
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the input image
    ///
    /// Image should be 8bit RGBA
    #[arg(short, long)]
    img_path: PathBuf,
    /// Iteration count when performing model estimation
    #[arg(short, long, default_value_t = 1000)] //TODO: skal 1k være default værdi?
    pnp_ransac_iter_count: u32,
    #[command(subcommand)]
    cam_matrix: CameraIntrinsic,
    /// Whether or not to run this program in demo mode, defaults to false
    #[arg(long, default_value_t = false)]
    demo: bool,
}

#[derive(Subcommand)]
enum CameraIntrinsic {
    /// Load camera parameters from a file
    File {
        ///path to a file containing necesarry parameters
        path: String,
    },
    /// Manually specify parameters, any optional values will be set to 0
    Manual {
        /// Focal length in pixel units
        focal_len_x: f64,
        /// Usually the same as `focal_len_x`
        focal_len_y: f64,
        /// Axis skew, defaults to 0
        #[arg(default_value_t = 0.0)]
        skew: f64,
        /// Principal offset x coordinate, defaults to 0
        #[arg(default_value_t = 0.0)]
        offset_x: f64,
        /// Principal offset y coordinate
        #[arg(default_value_t = 0.0)]
        offset_y: f64,
    },
}

#[allow(unreachable_code)]
#[forbid(clippy::unwrap_used)]
fn main() {
    // dotenv().expect("failed to read environment variables");
    let args = Args::parse();
    // 1/0f64.floor() as i32; // :)
    let path = args.img_path.to_str().expect("img_path is not valid unicode");

    let image = Cmat::<BGRA8>::imread_checked(path, -1).expect("Failed to read image ");

    let (keypoints, descriptors) = akaze_keypoint_descriptor_extraction_def(&image.mat)
        .expect("AKAZE keypoint extraction failed");

    let k: i32 = todo!();
    let filter_strength: f32 = todo!();
    let dmatches =
        get_knn_matches(&descriptors, todo!(), k, filter_strength).expect("knn matches failed");

    let point_correspondences = get_points_from_matches(&keypoints, todo!(), &dmatches)
        .expect("failed to obtain point correspondences");
    //TODO: map reference image keypoints to 3d coordinates
    let ref_kp_woorld_coords: Vec<opencv::core::Point3f> = point_correspondences
        .1
        .into_iter()
        .map(|f| todo!())
        .collect();

    //TODO: use ObjImgPointcorrespondence
    let point_correspondences: Vec<(Point2f, Point3f)> = point_correspondences
        .0
        .into_iter()
        .zip(ref_kp_woorld_coords)
        .map(|f| todo!())
        .collect();
    let camera_matrix = get_camera_matrix(args.cam_matrix).expect("Failed to get camera matrix");
}
