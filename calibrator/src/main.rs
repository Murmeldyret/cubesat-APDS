use std::{env::args, path::PathBuf};

use opencv::calib3d;
use clap::Parser;

#[derive(Parser)]
#[command(version,about,long_about=None)]
struct Args {

    /// Path to a directory containing calibration images
    img_path: PathBuf,
}

fn main() {
    
    let args = Args::parse();
    if args.img_path.is_dir() {

    }

}
