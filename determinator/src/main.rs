use clap::{Parser, Subcommand};
use dotenvy::dotenv;

use feature_database::keypointdb::KeypointDatabase;
use feature_database::models::Keypoint;
use feature_database::schema::keypoint::descriptor;
use feature_extraction::{
    akaze_keypoint_descriptor_extraction_def, get_knn_matches, get_points_from_matches,
};
use helpers::{
    get_camera_matrix, img_obj_corres, read_and_extract_kp, Coordinates3d, ReadAndExtractKpResult,
};
use homographier::homographier::{pnp_solver_ransac, raster_to_mat, Cmat, ImgObjCorrespondence};

use diesel::{Connection, PgConnection};
use opencv::calib3d::SolvePnPMethod;
use opencv::core::{Point2f, Point3_, Point3d, Point3f, Vector};
use rgb::alt::BGRA;
use rgb::{alt::BGRA8, RGBA};
use std::env;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use crate::helpers::{point2d_to_3d, point2f_to2d, point3f_to3d, project_obj_point, ref_keypoints};

pub mod helpers;

///TODO:
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the input image
    ///
    /// Image should be 8bit RGB, otherwise, opencv will try to convert
    #[arg(short, long)]
    img_path: PathBuf,
    /// TODO: @Murmeldyret
    #[arg(short, long)]
    lod: i32,
    /// Iteration count when performing model estimation
    #[arg(short, long, default_value_t = 1000)] //TODO: skal 1k være default værdi?
    pnp_ransac_iter_count: u32,
    #[command(subcommand)]
    cam_matrix: CameraIntrinsic,
    /// List of distortion coefficients, if any is supplied, length should be either 4,5,8,8,12 or 14
    #[arg(short,long,num_args(4..))]
    dist_coeff: Option<Vec<f64>>,
    /// Whether or not to run this program in demo mode
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
        /// Axis skew
        #[arg(default_value_t = 0.0)]
        skew: f64,
        /// Principal offset x coordinate
        #[arg(default_value_t = 0.0)]
        offset_x: f64,
        /// Principal offset y coordinate
        #[arg(default_value_t = 0.0)]
        offset_y: f64,
    },
}

type DbType = Arc<Mutex<PgConnection>>;
#[allow(unreachable_code)]
#[forbid(clippy::unwrap_used)]
fn main() {
    dotenv().expect("failed to read environment variables");
    let args = Args::parse();

    let extraction = read_and_extract_kp(&args.img_path);

    let point_correspondences = img_obj_corres(&args, extraction);
    let camera_matrix = get_camera_matrix(args.cam_matrix).expect("Failed to get camera matrix");

    // TODO: needs real camera matrix. Probably also a good guess for reproj_thres and confidence
    let solution = pnp_solver_ransac(
        &point_correspondences,
        &camera_matrix,
        args.pnp_ransac_iter_count.try_into().unwrap_or(1000),
        10000f32,
        0.9,
        args.dist_coeff.as_deref(),
        Some(SolvePnPMethod::SOLVEPNP_EPNP), // i think this method is most appropriate, optionally it could be program argument
    )
    .expect("Failed to solve PNP problem")
    .expect("No solution was obtained to the PNP problem");
    // dbg!(solution);
    // TODO: By using the obtained solution, we can map object points to image points, this must be done to find where the image points lie in 3d space
    // evt kig på dette: https://en.wikipedia.org/wiki/Perspective-n-Point#Methods
    project_obj_point(Point3d::new(0.0, 0.0, 0.0), solution, &camera_matrix);
}
