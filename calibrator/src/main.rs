use std::{path::PathBuf};

use clap::Parser;
use helpers::*;


pub mod helpers;
#[derive(Parser)]
#[command(version,about,long_about=None)]
struct Args {
    /// Path to a directory containing calibration images
    img_path: PathBuf,
}

fn main() {
    let args = Args::parse();
    let images = read_images(&args.img_path);
    assert!(
        images.len() >= 10,
        "At least 10 images are necesarry to perform image calibration, but only {} was found",
        images.len()
    );

    
}
