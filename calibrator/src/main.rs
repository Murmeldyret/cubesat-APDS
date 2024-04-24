use std::{iter, path::PathBuf};

use clap::Parser;
use helpers::*;
use homographier::homographier::Cmat;
use opencv::{
    calib3d::{calibrate_camera, calibrate_camera_def, draw_chessboard_corners, CALIB_CB_ADAPTIVE_THRESH},
    core::{MatTraitConst, Point2f, Point3f, Size2i, TermCriteria, TermCriteria_COUNT, TermCriteria_MAX_ITER, TermCriteria_Type, Vec3d, Vector, CV_64FC3},
    highgui::{destroy_all_windows, imshow, wait_key},
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
    let mut images = read_images(&args.img_path);
    assert!(
        images.len() >= 10,
        "At least 10 images are necesarry to perform image calibration, but only {} was found",
        images.len()
    );
    // dbg!(images.len());
    let size = opencv::core::Size::new(args.corners[0].into(), args.corners[1].into());
    let obj_points = iter::repeat(img_points_from_size(&size)).take(images.len()).collect::<Vector<Vector<_>>>();
    let mut corners: Vec<Vector<Point2f>> = Vec::with_capacity(images.len());
    // dbg!(corners);
    let mut corner_found: Vec<bool> = Vec::with_capacity(images.len());

    for (i, elem) in images.iter().enumerate() {
        let mut corner: Vector<Point2f> = Vector::new();
        let res = opencv::calib3d::find_chessboard_corners(
            &elem.mat,
            size.clone(),
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
    
    
    let res = calibrate_camera_def(
        &obj_points,
        &img_points,
        images[0].mat.size().expect("epic fail"),
        &mut cam_mat.mat,
        &mut dist_coeffs,
        &mut _rvec.mat,
        &mut _tvec.mat,
    );
    dbg!(res.unwrap());
    // dbg!(cam_mat.at_2d(0, 0));
    // dbg!(cam_mat.at_2d(1, 1));
    // dbg!(cam_mat.at_2d(0, 1));
    // dbg!(cam_mat.at_2d(2, 2));
    // dbg!(cam_mat.at_2d(0, 2));
    // dbg!(cam_mat.at_2d(1, 2));
    dbg!(dist_coeffs.len());
    // for (i, elem) in corners.iter().enumerate() {
    //     draw_chessboard_corners(&mut images[i].mat, size, elem, corner_found[i]).expect("msg");
    //     imshow("pattern", &images[i].mat).expect("epic fail");
    //     wait_key(5000).expect("msg");
    // }
    // destroy_all_windows().expect("msg");
}
