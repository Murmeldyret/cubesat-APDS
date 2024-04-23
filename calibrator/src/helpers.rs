use std::{ffi::OsStr, path::Path};

use homographier::homographier::Cmat;
use opencv::imgcodecs::IMREAD_GRAYSCALE;

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

fn valid_img_extension(ext: &OsStr) -> bool {
    matches!(ext.to_str(), Some(val) if ["png", "jpg", "jpeg", "tif", "tiff"].contains(&val))
}
