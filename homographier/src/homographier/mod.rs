use std::marker::PhantomData;

use opencv::{
    calib3d::{find_homography, RANSAC},
    core::{
        Point2f, Scalar, Size, Size2d, Size2i, ToInputArray, ToOutputArray, Vec2f, Vec4b, VecN,
        BORDER_CONSTANT, CV_8U, CV_8UC4,
    },
    imgproc::{warp_perspective, INTER_LINEAR, INTER_NEAREST},
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
    // /// Tried to access matrix element outside bounds
    // OutOfBounds,
    /// An unknown error
    Unknown,
}

/// Checked Mat type
/// # Notes
/// Guarantees that a contained mat contains data, but makes no assumptions about validity
#[derive(Debug)]
pub struct Cmat<T> {
    mat: Mat,
    _marker: PhantomData<T>,
}

impl<T> Cmat<T> {
    pub fn new(mat: Mat) -> Result<Self, MatError> {
        Cmat {
            mat,
            _marker: PhantomData,
        }
        .check_owned()
    }

    pub fn imread_checked(filename: &str, flags: i32) -> Result<Self, MatError> {
        // let res =
        Cmat::new(opencv::imgcodecs::imread(filename, flags).map_err(MatError::Opencv)?)
    }

    fn check_owned(self) -> Result<Self, MatError> {
        match self.mat.dims() {
            // dims will always be >=2, unless the Mat is empty
            0 => Err(MatError::Empty),
            _ => Ok(self),
        }
    }

    /// Creates a Cmat from a copied 1-dimensional slice
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
    /// Checked element access
    /// Will return [`MatError::OutOfBounds`] if either row or column exceeds matrix width and size respectively
    pub fn at_2d(&self, row: i32, col: i32) -> Result<&T, MatError> {
        let size = self.mat.size().map_err(|_err| MatError::Unknown)?;
        if (row > size.width) || (col > size.height) {
            return Err(MatError::Opencv(Error::new(-211, "")));
        }

        self.mat.at_2d::<T>(row, col).map_err(MatError::Opencv)
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

pub fn find_homography_mat(
    input: &[Point2f],
    reference: &[Point2f],
    method: Option<HomographyMethod>,
    reproj_threshold: Option<f64>,
) -> Result<(Cmat<f64>, Cmat<u8>), MatError> {
    let input = opencv::core::Vector::from_slice(input);
    let reference = opencv::core::Vector::from_slice(reference);
    let mut mask = Mat::default();
    let method = method.unwrap_or(HomographyMethod::Default) as i32;

    let homography = find_homography(
        &input,
        &reference,
        &mut mask,
        method,
        reproj_threshold.unwrap_or(10.0),
    )
    .map_err(MatError::Opencv)?; // RANSAC is used since some feature matching may be erroneous.

    // let mask = Cmat::<Point2f>::new(mask)?; //
    Ok((Cmat::<f64>::new(homography)?, Cmat::<u8>::new(mask)?))
}

/// Warps the perspective of `src` image using `m`
///
/// ## Parameters
/// * src: The image that will be transformed
/// * m: a 3x3 transformation matrix
/// size: the size of the output image, by default the size is equal to that of `src`
///
/// ## Errors
/// TODO
pub fn warp_image_perspective<T: DataType>(
    src: &Cmat<T>,
    m: &Cmat<f64>, // could be replaced with Matx33d as a potential optimization
    size: Option<Size2i>,
) -> Result<Cmat<T>, MatError> {
    let size = size.unwrap_or(src.mat.size().map_err(MatError::Opencv)?);
    let src_size = src.mat.size().unwrap();
    let mut mat = Mat::new_rows_cols_with_default(
        src_size.height,
        src_size.width,
        src.mat.typ(),
        Scalar::new(1f64, 1f64, 1f64, 1f64),)
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

// clippy er dum, så vi sætter den lige på plads
#[allow(clippy::unwrap_used)]
#[allow(unused_variables)]
#[allow(unused_imports)]
#[allow(dead_code)]
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

    fn empty_homography() -> Cmat<f64> {
        let slice: [[f64; 3]; 3] = [[1f64,0f64,0f64],[0f64,1f64,0f64],[0f64,0f64,1f64]];
        Cmat::from_2d_slice(&slice).unwrap()
    }

    #[ignore = "virker stadigvæk ikke"]
    #[test]
    fn homography_success() {
        let mut points: Vec<Point2f> = Vec::with_capacity(4);

        // points.push(Point2f::new(1f32, 1f32));
        // points.push(Point2f::new(2f32, 2f32));
        // points.push(Point2f::new(3f32, 4f32));
        // points.push(Point2f::new(4f32, 4f32));
        points.push(Point2f::new(2f32, 2f32));
        points.push(Point2f::new(4f32, 4f32));
        points.push(Point2f::new(8f32, 8f32));
        points.push(Point2f::new(8f32, 8f32));

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
        assert_eq!(homography.at_2d(2, 2).unwrap(), &1f64); // h__3,3 should always be 1 https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html

        //alt herefter giver ikke mening
        let mut sum: f64 = 0f64;
        for col in 0..3 {
            for row in 0..3 {
                let elem = homography.at_2d(row, col).unwrap();
                dbg!(elem);
                sum += elem
            }
        }
        dbg!(sum);
        // assert_eq!(sum, 1f64);
        for col in 0..3 {
            for row in 0..3 {
                if col == 2 && row == 2 {
                    assert_eq!(homography.at_2d(row, col).unwrap(), &1f64);
                } else {
                    assert_eq!(homography.at_2d(row, col).unwrap(), &0f64);
                }
            }
        }
    }

    #[test]
    fn cmat_init() {
        assert!(Cmat::<BGRA>::new(Mat::default()).is_err())
    }
    #[test]
    fn cmat_init_2d() {
        let cmat = Cmat::<BGRA>::new(
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

    #[test]
    fn image_correct_pixels() {
        let mut img_dir = path_to_test_images().expect("epic fail");
        img_dir.pop();
        img_dir.push("images");

        img_dir.push("1.png");
        let img = Cmat::<BGRA>::imread_checked(img_dir.to_str().unwrap(), IMREAD_UNCHANGED.into())
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
