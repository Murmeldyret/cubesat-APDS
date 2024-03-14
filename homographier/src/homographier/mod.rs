use opencv::{calib3d::{find_homography, prelude::*, RANSAC}, core::{InputArray, ToInputArray}, prelude::*};
use rgb::*;

fn raster_to_mat(pixels: &[RGB<f32>]) -> Mat {
    
    todo!()
}


fn find_homography_mat(input: &impl ToInputArray,reference: &impl ToInputArray) -> Result<Mat,opencv::Error> {
    let mut mask = Mat::default();
    const REPROJC_THRESHOLD: f64 = 10.0; // opencv siger mellem 1 og, kræver ekperimantion
    let homography = find_homography(input, reference, &mut mask, RANSAC, REPROJC_THRESHOLD);
    todo!()
}