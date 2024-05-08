use clap::{Parser, Subcommand};
use dotenvy::dotenv;

use helpers::{
    get_camera_matrix, img_obj_corres, read_and_extract_kp
};
use homographier::homographier::pnp_solver_ransac;

use diesel::PgConnection;
use opencv::calib3d::SolvePnPMethod;
use opencv::core::{MatTraitConst, Point3d};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use crate::helpers::project_obj_point;

pub mod helpers;

///TODO:
#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Args {
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
    #[arg(short,long,num_args(4..),allow_negative_numbers=true)]
    dist_coeff: Option<Vec<f64>>,
    /// Whether or not to run this program in demo mode
    #[arg(long)]
    demo: bool,
}

#[derive(Subcommand)]
pub enum CameraIntrinsic {
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
    if let Some(coeffs) = &args.dist_coeff {
        assert!(
            matches!(coeffs.len(), 4 | 5 | 8 | 12 | 14),
            "Distortion coefficient length does not have required length of 4|5|8|12|14, found {}",
            coeffs.len()
        );
    }
    let extraction = read_and_extract_kp(&args.img_path);

    let point_correspondences = img_obj_corres(&args, extraction);
    let camera_matrix = get_camera_matrix(args.cam_matrix).expect("Failed to get camera matrix");

    // TODO: needs real camera matrix. Probably also a good guess for reproj_thres and confidence
    let solution = pnp_solver_ransac(
        &point_correspondences,
        &camera_matrix,
        args.pnp_ransac_iter_count.try_into().unwrap_or(10000),
        8f32,
        1.0 - f64::EPSILON, /*TIHI*/
        args.dist_coeff.as_deref(),
        Some(SolvePnPMethod::SOLVEPNP_EPNP), // i think this method is most appropriate, optionally it could be program argument
    )
    .expect("Failed to solve PNP problem")
    .expect("No solution was obtained to the PNP problem");
    dbg!(solution.inliers.mat.size());
    // TODO: By using the obtained solution, we can map object points to image points, this must be done to find where the image points lie in 3d space
    // evt kig på dette: https://en.wikipedia.org/wiki/Perspective-n-Point#Methods
    println!(
        "rotation matrix =\n{}translation matrix =\n{}",
        solution.rvec.format_elems(),
        solution.tvec.format_elems()
    );
    println!(
        "Inlier ratio: {:.2} ({}/{})",
        (solution.inliers.mat.rows() as f64) / (point_correspondences.len() as f64),
        solution.inliers.mat.rows(),
        point_correspondences.len(),
    );
    dbg!(project_obj_point(
        Point3d::new(1.0, 1.0, 1.0),
        solution,
        camera_matrix
    ));
}
