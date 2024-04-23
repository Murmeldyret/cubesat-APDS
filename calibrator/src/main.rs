use std::path::PathBuf;

use clap::Parser;
use helpers::*;
use opencv::{
    calib3d::CALIB_CB_ADAPTIVE_THRESH,
    core::{Point2f, Vector},
};

pub mod helpers;
#[derive(Parser)]
#[command(
    version,
    about,
    long_about = "Find camera calibration matrix from input images"
)]
struct Args {
    /// Path to a directory containing calibration images
    #[arg(short, long)]
    img_path: PathBuf,
    /// Corners in the input pattern
    #[arg(short,long,num_args(2..3))]
    corners: Vec<u8>,
}

fn main() {
    let args = Args::parse();
    let images = read_images(&args.img_path);
    assert!(
        images.len() >= 10,
        "At least 10 images are necesarry to perform image calibration, but only {} was found",
        images.len()
    );
    let size = opencv::core::Size::new(args.corners[0].into(), args.corners[1].into());
    let mut corners: Vec<Vector<Point2f>> = Vec::with_capacity(images.len());

    for (i, elem) in images.iter().enumerate() {
        opencv::calib3d::find_chessboard_corners(&elem.mat, size, &mut corners[i], CALIB_CB_ADAPTIVE_THRESH).expect("Opencv erorr").then_some(()).expect("could not find chessboard corners in image, perhaps the image is not of a chessboard pattern");
    }
}
