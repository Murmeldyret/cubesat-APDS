use std::{ffi::OsStr, fs::DirEntry, path::PathBuf};

use homographier::homographier::Cmat;
use rgb::alt::BGRA8;

pub fn read_images(p: &PathBuf) -> Vec<Cmat<BGRA8>> {
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
        .filter_map(|f| f.path().to_str().map(|s| <&str as Into<String>>::into(s)))
        .collect::<Vec<String>>();

    todo!()
}

fn valid_img_extension(ext: &OsStr) -> bool {
    match ext.to_str() {
        Some(val) if ["png", "jpg", "jpeg", "tif", "tiff"].contains(&val) => true,
        _ => false,
    }
}
// fn read_if_image(e: DirEntry) -> Option<Cmat<BGRA8>> {
//     e.

//     todo!()
// }
