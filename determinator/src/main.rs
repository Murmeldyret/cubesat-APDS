use clap::Parser;
use dotenvy::dotenv;

use feature_database::schema::keypoint::descriptor;
use feature_extraction::akaze_keypoint_descriptor_extraction_def;
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
    /// Whether or not to run this program in demo mode
    #[arg(long, default_value_t = false)]
    demo: bool,
}

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

    
}
