use clap::Parser;
use dotenvy::dotenv;

use feature_database::models::Keypoint;
use feature_database::schema::keypoint::descriptor;
use feature_extraction::{
    akaze_keypoint_descriptor_extraction_def, get_knn_matches, get_points_from_matches,
};
use homographier::homographier::{raster_to_mat, Cmat};

use diesel::PgConnection;
use rgb::alt::BGRA;
use rgb::{alt::BGRA8, RGBA};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub mod helpers;

///TODO:
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the input image
    ///
    /// Image should be 8bit RGBA
    #[arg(short, long)]
    img_path: PathBuf,
    /// Iteration count when performing model estimation
    #[arg(short, long)]
    pnp_ransac_iter_count: Option<u32>,
    /// Whether or not to run this program in demo mode, defaults to false
    #[arg(long, default_value_t = false)]
    demo: bool,
}

#[forbid(clippy::unwrap_used)]
fn main() {
    // dotenv().expect("failed to read environment variables");
    let args = Args::parse();
    let path = args
        .img_path
        .to_str()
        .expect("failed to read img_path, it is probably not valid unicode");

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
}
