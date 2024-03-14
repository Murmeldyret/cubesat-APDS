use crate::image_extractor::{Datasets, RawDataset};
use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

pub mod image_extractor;

fn main() {
    let mut current_dir = env::current_dir().expect("Current directory not set.");

    current_dir.push("resources/test/Geotiff");

    let paths = fs::read_dir(&current_dir).unwrap();

    let paths_vec: Vec<String> = paths
        .into_iter()
        .map(|p| p.unwrap().path().to_string_lossy().into())
        .collect();

    let raw_dataset = RawDataset::import_datasets(&paths_vec).unwrap();

    let output_path = format!("{}/{}", current_dir.to_string_lossy(), "resources/dataset");

    let mosaic = raw_dataset.mosaic_datasets(&Path::new(output_path.as_str()));
}

