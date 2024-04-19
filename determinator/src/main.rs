use clap::{Parser, Subcommand};
use dotenvy::dotenv;

use feature_database::keypointdb::KeypointDatabase;
use feature_database::models::Keypoint;
use feature_database::schema::keypoint::descriptor;
use feature_extraction::{
    akaze_keypoint_descriptor_extraction_def, get_knn_matches, get_points_from_matches,
};
use helpers::{get_camera_matrix, read_and_extract_kp, Coordinates3d};
use homographier::homographier::{raster_to_mat, Cmat};

use diesel::{Connection, PgConnection};
use opencv::core::{Point2f, Point3f, Vector};
use rgb::alt::BGRA;
use rgb::{alt::BGRA8, RGBA};
use std::env;
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
    let conn: DbType = Arc::new(Mutex::new(
        Connection::establish(
            &env::var("DATABASE_URL").expect("Error reading environment variable"),
        )
        .expect("Failed to connect to database"),
    ));

    let (image, keypoints, descriptors) = read_and_extract_kp(args.img_path);

    let k: i32 = todo!();
    let filter_strength: f32 = todo!();
    let dmatches =
        get_knn_matches(&descriptors.mat, todo!(), k, filter_strength).expect("knn matches failed");
    


    let ref_keypoints = feature_database::keypointdb::Keypoint::read_keypoints_from_lod(
        &mut conn.lock().unwrap(),
        todo!("@Rasmus plz"),
    )
    .expect("epic db fail");

    
    let ref_keypoints: Vector<opencv::core::KeyPoint> =
        Vector::from_iter(ref_keypoints.into_iter().map(|f| {
            opencv::core::KeyPoint::new_point(
                Point2f::new(f.x_coord as f32, f.y_coord as f32),
                f.size as f32,
                f.angle as f32,
                f.response as f32,
                f.octave,
                f.class_id,
            )
            .expect("error in converting db keypoint to opencv keypoint")
        }));

    
    let (img_points, obj_points) = get_points_from_matches(&keypoints, &ref_keypoints, &dmatches)
        .expect("failed to obtain point correspondences");
    assert!(
        img_points.len() >= 4,
        "Image points length must be at least 4"
    );
    assert!(
        obj_points.len() >= 4,
        "Object points length must be at least 4"
    );
    //TODO: map reference image keypoints to 3d coordinates
    let ref_kp_woorld_coords: Vec<opencv::core::Point3f> =
        obj_points.into_iter().map(|f| todo!()).collect();

    //TODO: use ObjImgPointcorrespondence
    let point_correspondences: Vec<(Point2f, Point3f)> = img_points
        .into_iter()
        .zip(ref_kp_woorld_coords)
        .map(|f| todo!())
        .collect();
    let camera_matrix = get_camera_matrix(args.cam_matrix).expect("Failed to get camera matrix");
}
