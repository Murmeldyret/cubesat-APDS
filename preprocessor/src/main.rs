use dotenvy::dotenv;
use feature_database;
use feature_extraction;
use geotiff_lib::image_extractor;
use geotiff_lib::image_extractor::{Datasets, MosaicDataset, MosaicedDataset};

use clap::Parser;
use std::env;

pub mod level_of_detail;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path to the folder where datasets are stored on disk
    #[arg(short, long)]
    dataset_path: String,

    /// The path to the folder which shall contain processed files.
    #[arg(short, long, default_value_t = String::from("processed_files/"))]
    output_path: String,

    /// The database url to connect to. Can also be provided by setting environment variable: DATABASE_URL
    #[arg(long)]
    database_url: Option<String>,

    /// The tile size which the reference image will be split into
    #[arg(short, long, default_value_t = 1000)]
    tile_size: u64,
}

fn main() {
    dotenv().expect("Could not read .env file");

    let args = Args::parse();

    let db_connection = &mut feature_database::db_helpers::setup_database();

    let dataset = image_extractor::RawDataset::import_datasets(&args.dataset_path)
        .expect("Could not open datasets");

    let mosaic = dataset
        .to_mosaic_dataset(&args.output_path)
        .expect("Could not convert dataset.");

    // let levels_of_detail = level_of_detail::calculate_amount_of_levels(, TILE_RESOLUTION)
}

fn downscale_from_lod(image: MosaicedDataset, tile_size: u64, lod: u64) {
    let image_resolution = image.dataset.raster_size();

    let columns: u64 = image_resolution.0 as u64 / tile_size;
    let rows: u64 = image_resolution.1 as u64 / tile_size;

    for i in 0..rows {
        for j in 0..columns {
            let tile = image.to_rgb(
                ((i * tile_size) as isize, (j * tile_size) as isize),
                ((tile_size * lod) as usize, (tile_size * lod) as usize),
                (tile_size as usize, tile_size as usize),
            );
        }
    }
}
