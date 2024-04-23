use std::marker::PhantomData;

use opencv::{
    calib3d::{find_homography, solve_pnp_ransac, SolvePnPMethod, RANSAC},
    core::{
        Point2d, Point2f, Point3d, Scalar, Size2i, ToInputArray, ToOutputArray, Vec4b, Vector,
        BORDER_CONSTANT, CV_8UC4,
    },
    imgproc::{warp_perspective, INTER_LINEAR},
    prelude::*,
    Error,
};
use rgb::*;

pub trait PixelElemType {
    fn to_cv_const(&self) -> i32;
}
pub struct BGRA;
impl PixelElemType for BGRA {
    fn to_cv_const(&self) -> i32 {
        CV_8UC4
    }
}

#[derive(Clone, Copy)]
pub enum HomographyMethod {
    Default = 0,
    LMEDS = 4,
    RANSAC = 8,
    RHO = 16,
}

#[non_exhaustive]
#[derive(Debug)]
pub enum MatError {
    /// Inner openCV errors
    Opencv(opencv::Error),
    /// The mat is empty, and not considered valid
    Empty,
    /// Matrix is not rectangular (columns or rows with differing lengths)
    Jagged,
    /// An unknown error
    Unknown,
}

#[derive(Debug)]
pub struct PNPRANSACSolution {
    pub rvec: Cmat<f64>,
    pub tvec: Cmat<f64>,
    pub inliers: Cmat<i32>,
}
/// 3D object point and its corresponding 2d image point
pub struct ImgObjCorrespondence {
    pub obj_point: Point3d,
    pub img_point: Point2d,
}

impl ImgObjCorrespondence {
    pub fn new(obj_point: Point3d, img_point: Point2d) -> Self {
        ImgObjCorrespondence {
            obj_point,
            img_point,
        }
    }
}
/// Checked Mat type
/// T is the matrix element type, usually T should implement [`opencv::core::DataType`]
/// # Notes
/// Guarantees that a contained mat contains data, but makes no assumptions about validity
#[derive(Debug)]
pub struct Cmat<T> {
    pub mat: Mat,
    _marker: PhantomData<T>,
}

impl<T> Cmat<T> {
    fn from_mat(mat: Mat) -> Result<Self, MatError> {
        Cmat {
            mat,
            _marker: PhantomData,
        }
        .check_owned()
    }

    fn check_owned(self) -> Result<Self, MatError> {
        match self.mat.dims() {
            // dims will always be >=2, unless the Mat is empty
            0 => Err(MatError::Empty),
            _ => Ok(self),
        }
    }

    /// Creates a Cmat from a copied 2-dimensional slice
    pub fn from_2d_slice(slice: &[impl AsRef<[T]>]) -> Result<Self, MatError>
    where
        T: DataType,
    {
        let mat = Mat::from_slice_2d::<T>(slice).map_err(MatError::Opencv)?;
        Cmat::new(mat)
    }

    fn check(&self) -> Result<(), MatError> {
        match self.mat.dims() {
            // dims will always be >=2, unless the Mat is empty
            0 => Err(MatError::Empty),
            _ => Ok(()),
        }
    }

    //further checked functions go here
}

impl<T: DataType> Cmat<T> {
    pub fn new(mat: Mat) -> Result<Self, MatError> {
        match T::opencv_type() == mat.typ() {
            true => Ok(Cmat::from_mat(mat)?),
            false => Err(MatError::Empty),
        }
    }

    pub fn imread_checked(filename: &str, flags: i32) -> Result<Self, MatError> {
        // let res =
        Cmat::new(opencv::imgcodecs::imread(filename, flags).map_err(MatError::Opencv)?)
    }
    /// Checked element access
    ///
    /// ## Errors
    /// Will return [`MatError::OutOfBounds`] if either row or column exceeds matrix width and size respectively.
    /// If the type T does not match the inner Mat's type, an error is returned
    pub fn at_2d(&self, row: i32, col: i32) -> Result<&T, MatError> {
        let size = self.mat.size().map_err(|_err| MatError::Unknown)?;
        if (row > size.width) || (col > size.height) {
            return Err(MatError::Opencv(Error::new(-211, "")));
        }

        self.mat.at_2d::<T>(row, col).map_err(MatError::Opencv)
    }

    pub fn zeros(rows: i32, cols: i32) -> Result<Cmat<T>, MatError> {
        let mat = Mat::zeros(rows, cols, T::opencv_type())
            .map_err(MatError::Opencv)?
            .to_mat()
            .map_err(MatError::Opencv)?;
        Cmat::new(mat)
    }
}

impl<T> ToInputArray for Cmat<T> {
    fn input_array(&self) -> opencv::Result<opencv::core::_InputArray> {
        self.check().map_err(|err| match err {
            MatError::Opencv(inner) => inner,
            _ => opencv::Error {
                code: -2,
                message: "unknown error".into(),
            },
        })?;
        self.mat.input_array()
    }
}
impl<T> ToOutputArray for Cmat<T> {
    fn output_array(&mut self) -> opencv::Result<opencv::core::_OutputArray> {
        self.check().map_err(|err| match err {
            MatError::Opencv(inner) => inner,
            _ => opencv::Error {
                code: -2,
                message: "unknown error".into(),
            },
        })?;
        self.mat.output_array()
    }
}

/// Converts a slice of [`RGBA8`] to a [`Cmat<Vec4b>`]
///
/// ## Parameters
/// pixels: the slice of pixels that should be converted to a matrix, the slice length should be equal to `w*h`
/// w: the width of the image, or the number of columns
/// h: the height of the image, or the number of rows
/// ## Notes
/// Since OpenCV uses BGRA pixel ordering, the resulting matrix will be converted from RGBA to BGRA
/// ## Errors
/// Errors if pixel length != `w*h`
pub fn raster_to_mat(pixels: &[RGBA8], w: i32, h: i32) -> Result<Cmat<Vec4b>, MatError> {
    //RGBA<u8> is equivalent to opencv's Vec4b, which implements DataType
    if pixels.len() != (w * h) as usize {
        return Err(MatError::Unknown);
    }

    let rows = raster_1d_to_2d(pixels, w, None).map_err(|_err| MatError::Jagged)?;

    let converted: Vec<Vec<Vec4b>> = rows
        .into_iter()
        .map(|row| row.into_iter().map(rbga8_to_vec4b).collect())
        .collect();

    Cmat::from_2d_slice(&converted)
}

fn raster_1d_to_2d(
    pixels: &[RGBA8],
    w: i32,
    vec: Option<Vec<Vec<RGBA8>>>,
) -> Result<Vec<Vec<RGBA8>>, ()> {
    let (first_row, rest) = pixels.split_at(w as usize);
    let mut vec = vec.unwrap_or_default();

    vec.push(first_row.to_vec());

    let len = pixels.len() % (w as usize);

    match rest.len() {
        0 => Ok(vec),
        _ if len == 0 => raster_1d_to_2d(rest, w, Some(vec)),
        _ => Err(()), // if there is not enough pixels to fill a row
    }
}

fn rbga8_to_vec4b(pixel: RGBA8) -> Vec4b {
    Vec4b::new(pixel.b, pixel.g, pixel.r, pixel.a)
}

/// Estimates the homography between 2 planes, this matrix is always 3x3 `CV_F64C1`
/// ## Parameters
/// * input: Points taken from the source plane (length should be atleast 4, and points cannot be colinear)
/// * reference: Points taken from the destination plane (length should be atleast 4, and points cannot be colinear)
/// * method: the method used to compute, default is Least median squares (Lmeds)
/// * repreproj_threshold: Maximum allowed error, if the error is greater, a point is considered an outlier (used in RANSAC and RHO), defaults to 3.0
///
/// ## Errors
/// TODO
pub fn find_homography_mat(
    input: &[Point2f],
    reference: &[Point2f],
    method: Option<HomographyMethod>,
    reproj_threshold: Option<f64>,
) -> Result<(Cmat<f64>, Option<Cmat<u8>>), MatError> {
    let input = opencv::core::Vector::from_slice(input);
    let reference = opencv::core::Vector::from_slice(reference);

    let mut mask = Mat::default();
    let method_i = method.unwrap_or(HomographyMethod::Default) as i32;

    let homography = find_homography(
        &input,
        &reference,
        &mut mask,
        method_i,
        reproj_threshold.unwrap_or(3f64),
    )
    .map_err(MatError::Opencv)?; // RANSAC is used since some feature matching may be erroneous.

    // let mask = Cmat::<Point2f>::new(mask)?; //
    let out_mask = match method {
        Some(HomographyMethod::RANSAC) => Some(Cmat::new(mask)?),
        Some(HomographyMethod::LMEDS) => Some(Cmat::new(mask)?),
        _ => None,
    };
    Ok((Cmat::<f64>::new(homography)?, out_mask))
}

/// Warps the perspective of `src` image using `m`
///
/// ## Parameters
/// * src: The image that will be transformed
/// * m: a 3x3 transformation matrix, usually the one returned by [`find_homography_mat`]
/// size: the size of the output image, by default the size is equal to that of `src`
///
/// ## Errors
/// TODO
/// If the matrix m is not 3x3, an error is returned
pub fn warp_image_perspective<T: DataType>(
    src: &Cmat<T>,
    m: &Cmat<f64>, // could be replaced with Matx33d as a potential optimization
    size: Option<Size2i>,
) -> Result<Cmat<T>, MatError> {
    let size = size.unwrap_or(src.mat.size().map_err(MatError::Opencv)?);
    let src_size = src.mat.size().map_err(|_err| MatError::Unknown)?;
    let mut mat = Mat::new_rows_cols_with_default(
        src_size.height,
        src_size.width,
        src.mat.typ(),
        Scalar::new(1f64, 1f64, 1f64, 1f64),
    )
    .map_err(MatError::Opencv)?;

    warp_perspective(
        src,
        &mut mat,
        m,
        size,
        INTER_LINEAR,
        BORDER_CONSTANT,
        Scalar::new(1f64, 1f64, 1f64, 1f64),
    )
    .map_err(MatError::Opencv)?;

    debug_assert!(mat.typ() == src.mat.typ()); //stoler ikke på openCV

    Cmat::<T>::new(mat)
}

/// Estimates the pose of the camera using a subset of provided image-object point correspondences.
///
/// ## Parameters
/// * point_correspondences: a slice of 3d-to-2d point correspondences, minimum length is 4 (even in the P3P case, where the 4th point is used to find the solution with least reprojection error)
/// * camera_intrinsic: the camera calibration matrix 3X3
/// * iter_count: How many iteration the ransac algorithm should perform
/// * reproj_thres:
/// * confidence: //TODO
/// * dist_coeffs: distortion coefficients from camera calibration, if [`None`], a zero length vector is assumed
/// ## Returns
/// A solution, consisting of a rotation and translation matrix, and the indices of inliers used for the solution, returns `Ok(None)` if no solution was found
/// ## Errors
/// If the `point_correspondences` has less than 4 elements
/// # Notes
/// Since ransac randomly chooses a subset of points as the basis for a solution, the function behaves nondeterministiaclly.
/// As such there is no gurantee that produces the same solution with the same parameters
/// if the number of correspondence points is <=4, RANSAC will not be used.
/// The object points should not be colinear, if they are, a solution may not be found
pub fn pnp_solver_ransac(
    point_correspondences: &[ImgObjCorrespondence],
    camera_intrinsic: &Cmat<f64>,
    iter_count: i32,
    reproj_thres: f32,
    confidence: f64,
    dist_coeffs: Option<&[f64]>,
    method: Option<SolvePnPMethod>,
) -> Result<Option<PNPRANSACSolution>, MatError> {
    let (obj_points, img_points): (Vec<_>, Vec<_>) = point_correspondences
        .iter()
        .map(|p| (p.obj_point, p.img_point))
        .unzip();

    let obj_points = Vector::from_slice(&obj_points);
    let img_points = Vector::from_slice(&img_points);

    // output parameters
    let mut rvec = Cmat::<f64>::zeros(3, 1)?;

    let mut tvec = Cmat::<f64>::zeros(3, 1)?;

    let mut inliers = Cmat::<i32>::zeros(1, 1)?;

    let dist_coeffs = Cmat::<f64>::zeros(4, 1)?;

    // i think that Ok(false) means that there is no solution, but no errors happened
    let res = solve_pnp_ransac(
        &obj_points,
        &img_points,
        camera_intrinsic,
        &dist_coeffs,
        &mut rvec,
        &mut tvec,
        false,
        iter_count,
        reproj_thres,
        confidence,
        &mut inliers,
        method.unwrap_or(SolvePnPMethod::SOLVEPNP_EPNP) as i32,
    )
    .map_err(MatError::Opencv)?;
    let solution = PNPRANSACSolution {
        rvec,
        tvec,
        inliers,
    };
    let solution = res.then_some(solution);
    Ok(solution)
}
#[allow(unused_variables)]
#[allow(unused_imports)]
#[allow(dead_code)]
#[cfg(test)]
mod test {
    use crate::homographier::*;
    use opencv::{
        core::*,
        imgcodecs::{ImreadModes, IMREAD_UNCHANGED},
    };

    use rgb::alt::BGRA8;
    use std::{env, io, num::NonZeroIsize, path::PathBuf};
    type Image<T> = Vec<Vec<T>>;
    fn path_to_test_images() -> io::Result<PathBuf> {
        let mut img_dir = env::current_dir()?;

        img_dir.pop();
        img_dir.push("resources/test/images");
        Ok(img_dir)
    }

    fn test_image(size: usize) -> Cmat<Vec4b> {
        let pixel = RGBA8::new(1, 1, 1, 1);
        let image: Vec<RGBA8> = vec![pixel; size * size];
        let image: Vec<RGBA8> = image
            .into_iter()
            .enumerate()
            .map(|p| {
                let row = (p.0 / size) + 1;
                let col = (p.0 % size) + 1;
                let p = p.1;
                RGBA8::new(p.r, p.g * col as u8, p.b * row as u8, p.a)
            })
            .collect();
        let image = raster_to_mat(&image, size as i32, size as i32);
        image.unwrap()
    }

    fn camera_matrix() -> Cmat<f64> {
        let focal = (8.64f64, 8.64f64);
        let skew = 0f64;
        let principal_point = (0f64, 0f64);
        let s = vec![
            focal.0,
            skew,
            principal_point.0,
            focal.1,
            0f64,
            principal_point.1,
            0f64,
            0f64,
            1f64,
        ];
        let calib = Mat::from_slice_rows_cols(&s, 3, 3).unwrap();
        Cmat::<f64>::new(calib).unwrap()
    }

    const CAMERA_FOCAL_LENGTH_IN_MM: f64 = 16f64;

    fn empty_homography() -> Cmat<f64> {
        // an idempotent homography is also the identity matrix
        const SLICE: [[f64; 3]; 3] = [[1f64, 0f64, 0f64], [0f64, 1f64, 0f64], [0f64, 0f64, 1f64]];
        Cmat::from_2d_slice(&SLICE).unwrap()
    }

    #[test]
    fn homography_success() {
        let mut points: Vec<Point2f> = Vec::with_capacity(100);

        // Create many sample keypoints should be high, else it will distort the image too much.
        for i in 1..=10 {
            for j in 1..=10 {
                points.push(Point2f::new(i as f32, j as f32));
            }
        }

        let res = find_homography_mat(
            &points.clone(),
            &points.clone(),
            Some(HomographyMethod::RANSAC),
            Some(1f64),
        );
        let res = res.inspect_err(|e| {
            dbg!(e);
        });
        assert!(res.is_ok());

        let res = res.unwrap();
        let homography = res.0;
        let mask = res.1;

        // Assert for identity matrix
        for col in 0..3 {
            for row in 0..3 {
                if col == row {
                    assert_eq!(&homography.at_2d(row, col).unwrap().round(), &1f64);
                } else {
                    assert_eq!(&homography.at_2d(row, col).unwrap().round(), &0f64);
                }
            }
        }
    }

    #[test]
    fn cmat_init() {
        assert!(Cmat::<BGRA8>::new(Mat::default()).is_err())
    }
    #[test]
    fn cmat_init_2d() {
        let cmat = Cmat::<BGRA8>::new(
            Mat::new_size_with_default(Size::new(10, 10), CV_8UC4, Scalar::default()).unwrap(),
        )
        .unwrap();

        assert!(cmat.mat.dims() == 2)
    }

    #[test]
    #[ignore]
    fn mat_ones() {
        let mat: Mat = Mat::ones(2, 2, CV_8UC4).unwrap().to_mat().unwrap();
        // dbg!(mat.at_2d::<i32>(1, 1).unwrap());
        // assert_eq!(*mat.at_3d::<Vec4b>(0, 0, 0).unwrap(),1)
        // let bgramat: Mat4b = Mat4b::try_from(mat).unwrap();
        // let pixels = bgramat.at_2d::<Vec4b>(1, 1).unwrap();
        // pixels.as_bgra();
        // mat.at_2d::<Vec4b>(1, 1).unwrap()
    }
    #[ignore = "TODO"]
    #[test]
    fn image_correct_pixels() {
        let mut img_dir = path_to_test_images().expect("epic fail");
        img_dir.pop();
        img_dir.push("images");

        img_dir.push("1.png");
        let img = Cmat::<BGRA8>::imread_checked(img_dir.to_str().unwrap(), IMREAD_UNCHANGED.into())
            .expect("could not find image at location");

        assert_eq!(img.mat.depth(), CV_8U);
        assert_eq!(img.mat.channels(), 4);
    }

    #[test]
    fn cmat_from_slice() {
        let pixel: Vec4b = Vec4b::new(1, 2, 3, 4);
        let mut image: Image<Vec4b> = Vec::new();
        const IMG_SIZE: usize = 4;

        //matrix init
        for i in 0..IMG_SIZE {
            image.insert(i, Vec::new());
            image[i].reserve(IMG_SIZE);

            for j in 0..IMG_SIZE {
                let scalar: u8 = 1 + j as u8;
                image[i].insert(
                    j,
                    pixel
                        .clone()
                        .mul(Vec4b::new(scalar, scalar, scalar, scalar)),
                );
            }
        }
        let cmat = Cmat::from_2d_slice(&image).unwrap();
        let first_pixel = Vec4b::new(1, 2, 3, 4);
        let sixteenth_pixel = first_pixel.mul(Vec4b::new(4, 4, 4, 4));

        // asserts that pixels are stored row major i.e.
        // [[<1,2,3,4>,<2,4,6,8>,<3,6,9,12>,<4,8,12,16>],
        // [[<1,2,3,4>,<2,4,6,8>,<3,6,9,12>,<4,8,12,16>],
        // [[<1,2,3,4>,<2,4,6,8>,<3,6,9,12>,<4,8,12,16>],
        // [[<1,2,3,4>,<2,4,6,8>,<3,6,9,12>,<4,8,12,16>]]

        assert_eq!(cmat.mat.at_2d::<Vec4b>(0, 0).unwrap().clone(), first_pixel);
        assert_eq!(
            cmat.mat
                .at_2d::<Vec4b>((IMG_SIZE as i32) - 1, (IMG_SIZE as i32) - 1)
                .unwrap()
                .clone(),
            sixteenth_pixel
        );
    }

    #[test]
    fn raster_to_mat_works() {
        const IMG_SIZE: usize = 4;
        let pixel = RGBA8::new(1, 1, 1, 1);
        let image: Vec<RGBA8> = vec![pixel; IMG_SIZE * IMG_SIZE];
        let image: Vec<RGBA8> = image
            .into_iter()
            .enumerate()
            .map(|p| {
                let row = (p.0 / IMG_SIZE) + 1;
                let col = (p.0 % IMG_SIZE) + 1;
                let p = p.1;
                RGBA8::new(p.r, p.g * col as u8, p.b * row as u8, p.a)
            })
            .collect();
        let image = raster_to_mat(&image, IMG_SIZE as i32, IMG_SIZE as i32);

        assert!(image.is_ok());
        let image = image.unwrap();
        // assumes BGRA AND row major ordering
        assert_eq!(
            image.mat.at_2d::<Vec4b>(0, 0).unwrap().clone(),
            Vec4b::new(1, 1, 1, 1)
        );
        assert_eq!(
            image
                .mat
                .at_2d::<Vec4b>((IMG_SIZE - 1) as i32, (IMG_SIZE - 1) as i32)
                .unwrap()
                .clone(),
            Vec4b::new(IMG_SIZE as u8, IMG_SIZE as u8, 1, 1)
        );
        assert_eq!(
            image
                .mat
                .at_2d::<Vec4b>(0, (IMG_SIZE - 1) as i32)
                .unwrap()
                .clone(),
            Vec4b::new(1, IMG_SIZE as u8, 1, 1)
        );
        assert_eq!(
            image
                .mat
                .at_2d::<Vec4b>((IMG_SIZE - 1) as i32, 0)
                .unwrap()
                .clone(),
            Vec4b::new(IMG_SIZE as u8, 1, 1, 1)
        )
    }

    #[test]
    fn cmat_at_2d_works() {
        const IMG_SIZE: usize = 4;
        let image = test_image(IMG_SIZE);

        assert!(image
            .at_2d(3, 5)
            .is_err_and(|x| if let MatError::Opencv(e) = x {
                e.code_as_enum() == Some(Code::StsOutOfRange)
            } else {
                false
            }));
        assert!(image
            .at_2d(5, 3)
            .is_err_and(|x| if let MatError::Opencv(e) = x {
                e.code_as_enum() == Some(Code::StsOutOfRange)
            } else {
                false
            }));
        assert_eq!(image.at_2d(3, 3).unwrap().clone(), Vec4b::new(4, 4, 1, 1));
    }

    #[test]
    fn pnp_solver_ransac_no_work_lthan_3_points() {
        let corres_1 =
            ImgObjCorrespondence::new(Point3d::new(1f64, 2f64, 3f64), Point2d::new(1f64, 2f64));
        let corres_2 =
            ImgObjCorrespondence::new(Point3d::new(4f64, 5f64, 6f64), Point2d::new(4f64, 5f64));
        let corres_v = vec![corres_1, corres_2];
        let camera_intrinsic = Cmat::<f64>::zeros(3, 3).unwrap();
        let res = pnp_solver_ransac(&corres_v, &camera_intrinsic, 50, 2.0, 0.99, None, None);

        assert!(res.is_err(), "{:?}", res);
    }

    #[ignore = "Skal bruge Akaze keypoints"]
    #[test]
    fn pnp_solver_works() {
        let corres_1 = ImgObjCorrespondence::new(
            Point3d::new(0f64, 5f64, 1f64),
            Point2d::new(-1.48f64, 0.39f64),
        );
        let corres_2 = ImgObjCorrespondence::new(
            Point3d::new(5f64, 0f64, 0f64),
            Point2d::new(2.14f64, -1.92f64),
        );
        let corres_3 = ImgObjCorrespondence::new(
            Point3d::new(5f64, 5f64, 1.5f64),
            Point2d::new(1.74f64, 0.56f64),
        );
        let corres_4 = ImgObjCorrespondence::new(
            Point3d::new(0f64, 0f64, 1f64),
            Point2d::new(-2f64, -1.62f64),
        );
        let corres_5 = ImgObjCorrespondence::new(
            Point3d::new(2f64, 8f64, -2f64),
            Point2d::new(-0.16f64, 0.3f64),
        );

        let corres_v = vec![corres_1, corres_2, corres_3, corres_4, corres_5];
        let camera_intrinsic = camera_matrix();

        let res = pnp_solver_ransac(
            &corres_v,
            &camera_intrinsic,
            10000,
            100.0,
            0.5,
            None,
            Some(SolvePnPMethod::SOLVEPNP_P3P),
        );
        // no errors during solving
        assert!(res.is_ok(), "{:?}", res);
        let res = res.unwrap();
        assert!(res.is_some(), "No solution was found to the PNP problem");
        let res = res.unwrap();
    }
    #[test]
    fn warp_image_empty() {
        const SIZE: i32 = 4;
        let image = test_image(SIZE as usize);
        let homography = empty_homography();

        let warped = warp_image_perspective(&image, &homography, None);

        assert!(warped.is_ok());
        let warped = warped.expect("lol, lmao even");

        // applying an "empty" transformation should be idempotent
        for row in 0..SIZE {
            for col in 0..SIZE {
                print!("src: {:?} ", image.at_2d(row, col).unwrap());
                print!("dst: {:?}\n", warped.at_2d(row, col).unwrap());
                assert_eq!(
                    image.at_2d(row, col).unwrap(),
                    warped.at_2d(row, col).unwrap(),
                    "row: {} col: {}",
                    row,
                    col
                );
            }
        }
    }
}
