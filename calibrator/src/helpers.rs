use std::{ffi::OsStr, path::Path};

use homographier::homographier::Cmat;
use opencv::{
    core::{Point3f, Size2i, Vector},
    imgcodecs::IMREAD_GRAYSCALE,
};

/// Reads all valid images found in a provided path
/// ## Notes
/// The input path should be a directory
pub fn read_images(p: &Path) -> Vec<Cmat<u8>> {
    let res = p
        .read_dir()
        .expect("Failed to read input path")
        .filter_map(|f| f.ok())
        .filter(|p| {
            p.path()
                .extension()
                .filter(|ext| valid_img_extension(ext))
                .is_some()
        })
        .filter_map(|f| f.path().to_str().map(<&str as Into<String>>::into))
        .map(|f| {
            Cmat::<u8>::imread_checked(&f, IMREAD_GRAYSCALE)
                .unwrap_or_else(|_| panic!("Failed to read image named {}", f))
        })
        .collect::<Vec<Cmat<u8>>>();

    res
}

/// Determines whether a file extension is valid for opencv to read as image
fn valid_img_extension(ext: &OsStr) -> bool {
    matches!(ext.to_str(), Some(val) if ["png", "jpg", "jpeg", "tif", "tiff"].contains(&val.to_lowercase().as_str()))
}

pub fn img_points_from_size(size: &Size2i) -> Vector<Point3f> {
    let mut output: Vec<Point3f> = Vec::with_capacity(size.width as usize * size.height as usize);
    for i in 1..=size.width {
        for j in 1..=size.height {
            output.push(Point3f::new(i as f32, j as f32, 0f32));
        }
    }
    Vector::from_iter(output)
}
