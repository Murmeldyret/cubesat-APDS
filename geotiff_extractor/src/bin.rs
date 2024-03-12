use crate::image_extractor::{Datasets, RawDataset};
use std::env;
use std::fs;

pub mod image_extractor;

fn main() {
    // let mut current_dir = env::current_dir().expect("Current directory not set.");

    // current_dir.push("ressources/test/Geotiff");

    // let paths = fs::read_dir(current_dir).unwrap();

    // let paths_vec: Vec<String> = paths
    //     .into_iter()
    //     .map(|p| p.unwrap().path().display().to_string())
    //     .collect();

    // let paths: Vec<&str> = paths_vec.iter().map(AsRef::as_ref).collect();

    // let raw_dataset = RawDataset::import_datasets(&paths).unwrap();

    // let mosaic = raw_dataset.mosaic_datasets();
}

