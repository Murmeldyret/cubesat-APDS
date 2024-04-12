use diesel::PgConnection;
use dotenvy::dotenv;
use feature_database::imagedb::ImageDatabase;
use feature_database::{imagedb, keypointdb};
use feature_database::models;
use feature_extraction::akaze_keypoint_descriptor_extraction_def;
use geotiff_lib::image_extractor;
use geotiff_lib::image_extractor::{Datasets, MosaicDataset, MosaicedDataset};
use homographier::homographier::raster_to_mat;
use homographier::homographier::Cmat;

use level_of_detail::calculate_amount_of_levels;
use once_cell::sync::Lazy;
use raycon::ThreadPool;
use rayon as raycon;

use clap::Parser;
use rgb::RGBA;
use std::env;
use std::os::unix::thread;
use std::sync::{Arc, Mutex};

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

    /// The number of CPU threads to use for processing
    #[arg(short, long, default_value_t = 1)]
    cpu_num: usize,
}

type DbType = Arc<Mutex<PgConnection>>;

fn main() {
    dotenv().expect("Could not read .env file");

    let args = Args::parse();

    raycon::ThreadPoolBuilder::new().num_threads(args.cpu_num).build_global().unwrap();

    let db_connection: Arc<Mutex<PgConnection>> = Arc::new(Mutex::new(feature_database::db_helpers::setup_database()));

    println!("Read dataset");

    let dataset = image_extractor::RawDataset::import_datasets(&args.dataset_path)
        .expect("Could not open datasets");

    println!("Converting dataset to mosaic");

    let mosaic = dataset
        .to_mosaic_dataset(&args.output_path)
        .expect("Could not convert dataset.");

    println!("Processing mosaic");

    process_lod_from_mosaic(db_connection, mosaic, args.tile_size, args.cpu_num);
}

fn process_lod_from_mosaic(conn: DbType, image: MosaicedDataset, tile_size: u64, cpu_num: usize) {
    let image_resolution = image.get_dimensions().expect("Could not read image resolution");

    let amount_of_lod = calculate_amount_of_levels((image_resolution.0 * image_resolution.1) as u64, tile_size * tile_size);

    println!("Amount of lod: {}", &amount_of_lod);

    for i in 0..amount_of_lod {
        println!("Processing lod level: {}", &i);
        println!("=======================");
        downscale_from_lod(conn.clone(), &image, tile_size, i, cpu_num)
    }

}

fn downscale_from_lod(conn: DbType, image: &MosaicedDataset, tile_size: u64, lod: u64, cpu_num: usize) {
    
    let thread_pool = raycon::ThreadPoolBuilder::default().build().unwrap();

    let image_resolution = image.dataset.raster_size();

    let columns: u64 = image_resolution.0 as u64 / (tile_size * 2_i32.pow(lod as u32) as u64);
    let rows: u64 = image_resolution.1 as u64 / (tile_size * 2_i32.pow(lod as u32) as u64);

    println!("Columns: {}", &columns);
    println!("Rows: {}", &rows);

    for i in 0..rows {
        for j in 0..columns {
            println!("Processing Row: {}, Column: {}", &i, &j);
            let tile = image.to_rgb(
                ((i * tile_size) as isize, (j * tile_size) as isize),
                ((tile_size * lod) as usize, (tile_size * lod) as usize),
                (tile_size as usize, tile_size as usize),
            ).expect("Could not read tile from reference image");

            thread_pool.install(|| feature_extraction_to_database(conn.clone(), tile, tile_size, j, i, lod));
        }
    }
}

fn feature_extraction_to_database(conn: DbType, tile: Vec<RGBA<u8>>, tile_size: u64, column: u64, row: u64, lod: u64) {
    let tile_mat = raster_to_mat(&tile, tile_size as i32, tile_size as i32).expect("Could not convert tile to mat");

    
    let keypoints = akaze_keypoint_descriptor_extraction_def(&tile_mat.mat).unwrap().0.to_vec();
    
    let insert_image = models::InsertImage {
        level_of_detail: &(lod as i32),
        x_start: &((column * tile_size) as i32),
        x_end: &((column * tile_size + tile_size - 1) as i32),
        y_start: &((row * tile_size) as i32),
        y_end: &((row * tile_size + tile_size - 1) as i32)
    };
    
    let insert_image = imagedb::Image::One(insert_image);
    
    let mut conn = conn.lock().unwrap();

    imagedb::Image::create_image(&mut conn, insert_image).unwrap();

}