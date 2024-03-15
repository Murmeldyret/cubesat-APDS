use std::env;

use opencv::{
    calib3d::{find_homography, prelude::*, RANSAC},
    core::{InputArray, Scalar, Size, ToInputArray, ToOutputArray, CV_32FC4, CV_8UC4},
    imgcodecs::{ImreadModes, IMREAD_COLOR},
    prelude::*,
};
use rgb::*;

#[non_exhaustive]
#[derive(Debug)]
pub enum MatError {
    Opencv(opencv::Error),
    Empty,
    Unknown,
}

/// Checked Mat type
/// # Notes
/// Guarantees that a contained mat contains data, but makes no assumptions about validity
#[derive(Debug)]
pub struct Cmat(Mat);
impl Cmat {
    pub fn new(mat: Mat)->Result<Self,MatError> {
        Cmat(mat).check_owned()
    }
    pub fn imread_checked(filename: &str, flags: i32) -> Result<Self, MatError> {
        let res =
            Cmat(opencv::imgcodecs::imread(&filename, flags).map_err(|err| MatError::Opencv(err))?);

        res.check_owned()
    }
    fn check_owned(self) -> Result<Self, MatError> {
        match self.0.dims() { // dims will always be >=2, unless the Mat is empty
            0 => Err(MatError::Empty),
            _ => Ok(self),
        }
    }
    fn check(&self) -> Result<Self, MatError> {
        todo!()
    }

    //further checked functions go here
}
impl ToInputArray for Cmat {
    fn input_array(&self) -> opencv::Result<opencv::core::_InputArray> {
        let res = self.check().map_err(|err| match err {
            MatError::Opencv(inner) => inner,
            _ => opencv::Error {
                code: -2,
                message: "unknown error".into(),
            },
        })?;
        res.input_array()
    }
}
impl ToOutputArray for Cmat {
    fn output_array(&mut self) -> opencv::Result<opencv::core::_OutputArray> {
        self.check()
            .map_err(|err| match err {
                MatError::Opencv(inner) => inner,
                _ => opencv::Error {
                    code: -2,
                    message: "unknown error".into(),
                },
            })?
            .output_array()
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
#[ignore = "virker ikke helt endnu"]
#[test]
fn homography_success() {
    let mut img_dir = env::current_dir().expect("Current directory not set.");
    img_dir.pop();
    img_dir.push("images");
    // dbg!(current_dir);

    let mut input_path = img_dir.clone();
    input_path.push("3.png");
    let mut reference_path = img_dir.clone();
    reference_path.push("1.png");

    let input = opencv::imgcodecs::imread(
        input_path.to_str().unwrap(),
        ImreadModes::IMREAD_UNCHANGED.into(),
    )
    .unwrap();
    let reference = opencv::imgcodecs::imread(
        reference_path.to_str().unwrap(),
        ImreadModes::IMREAD_UNCHANGED.into(),
    )
    .unwrap();
    dbg!(&input);
    dbg!(&reference);
    let res = find_homography_mat(&input, &reference, None);
    let res = res.inspect_err(|e| {
        dbg!(e);
    });
    assert!(res.is_ok())
}
#[test]
fn cmat_init() {
    assert!(Cmat::new(Mat::default()).is_err())
}
#[test]
fn cmat_init_2d() {
    let cmat = Cmat::new(Mat::new_size_with_default(Size::new(10, 10), CV_8UC4, Scalar::default()).unwrap()).unwrap();
    
    assert!(cmat.0.dims()==2)
}
