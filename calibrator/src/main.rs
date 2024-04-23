use std::{env::args, path::PathBuf};

use clap::Parser;
use helpers::*;
use opencv::calib3d;

pub mod helpers;
#[derive(Parser)]
#[command(version,about,long_about=None)]
struct Args {
    /// Path to a directory containing calibration images
    img_path: PathBuf,
}

fn main() {
    let args = Args::parse();
}
