use std::{iter, path::PathBuf};

use clap::Parser;
use helpers::*;
use homographier::homographier::Cmat;
use opencv::{
    calib3d::{calibrate_camera_def, CALIB_CB_ADAPTIVE_THRESH},
    core::{MatTraitConst, Point2f, Vec3d, Vector},
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
    // dbg!(images.len());
    let size = opencv::core::Size::new(args.corners[0].into(), args.corners[1].into());
    let obj_points = iter::repeat(img_points_from_size(&size))
        .take(images.len())
        .collect::<Vector<Vector<_>>>();
    let mut corners: Vec<Vector<Point2f>> = Vec::with_capacity(images.len());
    // dbg!(corners);
    let mut corner_found: Vec<bool> = Vec::with_capacity(images.len());

    for (i, elem) in images.iter().enumerate() {
        let mut corner: Vector<Point2f> = Vector::new();
        let res = opencv::calib3d::find_chessboard_corners(
            &elem.mat,
            size,
            &mut corner,
            CALIB_CB_ADAPTIVE_THRESH,
        )
        .expect("Opencv error");
        corners.push(corner);
        corner_found.insert(i, res);
    }
    let img_points: Vector<Vector<opencv::core::Point_<f32>>> = Vector::from_iter(corners);

    let mut cam_mat = Cmat::<f64>::zeros(3, 3).expect("matrix intiliaztion should not fail");
    let mut dist_coeffs = Vector::<f64>::new();

    // these parameters does not really matter, but opencv wants them
    let mut _rvec = Cmat::<Vec3d>::zeros(3, 3).expect("TODO");
    let mut _tvec = Cmat::<Vec3d>::zeros(3, 3).expect("TODO");

    let rms_reproj = calibrate_camera_def(
        &obj_points,
        &img_points,
        images[0].mat.size().expect("epic fail"),
        &mut cam_mat.mat,
        &mut dist_coeffs,
        &mut _rvec.mat,
        &mut _tvec.mat,
    )
    .expect("Camera calibration estimation failed");
    let foc_x = cam_mat.at_2d(0, 0).expect("TODO");
    let skew = cam_mat.at_2d(0, 1).expect("TODO");
    let princip_x = cam_mat.at_2d(0, 2).expect("TODO");
    let foc_y = cam_mat.at_2d(1, 1).expect("TODO");
    let princip_y = cam_mat.at_2d(1, 2).expect("TODO");
    println!("|{:.3},{:.3},{:.3}|\n|0.000,{:.3},{:.3}|\n|0.000,0.000,1.000|\nRMS reprojection error:{:.3}",foc_x,skew,princip_x,foc_y,princip_y,rms_reproj);
    println!(
        "distortion coefficients: \n{:.?}",
        dist_coeffs.into_iter().collect::<Vec<f64>>()
    )
}
