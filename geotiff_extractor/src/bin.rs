use crate::image_extractor::RawDataset;
use std::env;
use std::fs;

pub mod image_extractor;

fn main() {
    let mut current_dir = env::current_dir().expect("Current directory not set.");

    current_dir.push("ressources/test/Geotiff");

    let paths = fs::read_dir(current_dir).unwrap();

    let paths_vec: Vec<&str> = paths
        .into_iter()
        .map(|p| p.unwrap().path().display().to_string().as_str())
        .collect();

    let something = image_extractor::RawDataset::import_datasets(&paths_vec).unwrap();
}

