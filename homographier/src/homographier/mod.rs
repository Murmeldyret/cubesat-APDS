use opencv::{
    calib3d::{find_homography, prelude::*, RANSAC},
    core::{InputArray, ToInputArray},
    imgcodecs::{ImreadModes, IMREAD_COLOR},
    prelude::*,
};
use rgb::*;
use std::boxed::Box;

/// checked Mat type
/// # Notes
/// Assumes that a 0 dimension Mat is erroneous, 
pub struct Cmat (Mat);
impl Cmat {
    pub fn imread_checked(filename: &str, flags: i32)-> Result<Self, ()> {
        let res = opencv::imgcodecs::imread(&filename, flags).map_err(|err|())?;

        todo!()
    }
}

fn raster_to_mat(pixels: &[RGB<f32>]) -> Mat {
    todo!()
}

fn find_homography_mat(
    input: &impl ToInputArray,
    reference: &impl ToInputArray,
    reproj_threshold: Option<f64>,
) -> Result<Mat, opencv::Error> {
    let mut mask = Mat::default();
    let homography = find_homography(
        input,
        reference,
        &mut mask,
        RANSAC,
        reproj_threshold.unwrap_or(10.0),
    ); // RANSAC is used since some feature matching may be erroneous.

    homography
}

#[test]
fn homography_success() {
    let input = opencv::imgcodecs::imread("./billed.png", ImreadModes::IMREAD_COLOR.into()).unwrap();
    let reference = opencv::imgcodecs::imread("./billed.png", ImreadModes::IMREAD_COLOR.into()).unwrap();
    dbg!(&input);
    let res = find_homography_mat(&input, &reference,None);

    assert!(res.is_ok())
}
