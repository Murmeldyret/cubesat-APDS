use diesel::PgConnection;
use dotenvy::dotenv;
use feature_database::models;
use feature_database::{imagedb, keypointdb};
use feature_database::{imagedb::ImageDatabase, keypointdb::KeypointDatabase};
use feature_extraction::{akaze_keypoint_descriptor_extraction_def, DbKeypoints};
use geotiff_lib::image_extractor;
use geotiff_lib::image_extractor::{Datasets, MosaicDataset, MosaicedDataset};
use homographier::homographier::raster_to_mat;
use homographier::homographier::Cmat;
use indicatif::{MultiProgress, ProgressBar};
use tempfile::tempdir;

use level_of_detail::calculate_amount_of_levels;
use once_cell::sync::Lazy;
use raycon::{Scope, ThreadPool};
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
    dataset_path: Option<String>,

    /// The path to a precomputed mosaic
    #[arg(short, long)]
    mosaic_path: Option<String>,

    /// The path to the temp folder which shall contain processed files (/tmp will be used if not provided)
    #[arg(long)]
    temp_path: Option<String>,

    /// The database url to connect to. Can also be provided by setting environment variable: DATABASE_URL
    #[arg(long)]
    database_url: Option<String>,

    /// The tile size which the reference image will be split into
    #[arg(short, long, default_value_t = 1000)]
    tile_size: u64,

    /// The number of CPU threads to use for processingcarg
    #[arg(short, long, default_value_t = 1)]
    cpu_num: usize,
}

type DbType = Arc<Mutex<PgConnection>>;

fn main() {
    dotenv().expect("Could not read .env file");

    let args = Args::parse();

    if args.dataset_path.is_none() && args.mosaic_path.is_none() {
        println!("No path provided to a dataset.");
        std::process::exit(1);
    }

    let db_connection: Arc<Mutex<PgConnection>> =
        Arc::new(Mutex::new(feature_database::db_helpers::setup_database()));

    println!("Read dataset");

    let thread_pool = raycon::ThreadPoolBuilder::new().num_threads(args.cpu_num).build().unwrap();

    thread_pool.scope(move |s| {
        let mosaic: Arc<Mutex<MosaicedDataset>>;
        if args.dataset_path.is_some() {
            let temp_dir = tempdir().expect(
                "Could not create temp directory\nPlease provide directory in the parameters",
            );
            let temp_path = args
                .temp_path
                .unwrap_or(temp_dir.into_path().to_string_lossy().to_string());
            let dataset = image_extractor::RawDataset::import_datasets(
                &args.dataset_path.expect("No path provided"),
            )
            .expect("Could not open datasets");
            println!("Converting dataset to mosaic");
            mosaic = Arc::new(Mutex::new(
                dataset
                    .to_mosaic_dataset(&temp_path)
                    .expect("Could not convert dataset."),
            ));
        } else if args.mosaic_path.is_some() {
            mosaic = Arc::new(Mutex::new(
                MosaicedDataset::import_mosaic_dataset(
                    &args.mosaic_path.expect("Expected mosaic path"),
                )
                .expect("Could not read mosaic"),
            ));
        } else {
            panic!("No dataset path provided");
        }

        println!("Processing mosaic");

        process_lod_from_mosaic(db_connection, mosaic, args.tile_size, s);
    });
}

fn process_lod_from_mosaic(
    conn: DbType,
    image: Arc<Mutex<MosaicedDataset>>,
    tile_size: u64,
    s: &Scope,
) {
    // let thread_pool = raycon::ThreadPoolBuilder::default().build().unwrap();

    let image_resolution = image
        .lock()
        .unwrap()
        .get_dimensions()
        .expect("Could not read image resolution");

    let amount_of_lod = calculate_amount_of_levels(
        (image_resolution.0 * image_resolution.1) as u64,
        tile_size * tile_size,
    );

    println!("Amount of lod: {}", &amount_of_lod);

    let multi_bar = MultiProgress::new();
    multi_bar
        .println(format!("Processing {} level of detail", amount_of_lod))
        .unwrap();

    for i in 0..amount_of_lod {
        downscale_from_lod(
            conn.clone(),
            image.clone(),
            tile_size,
            i,
            multi_bar.clone(),
            s,
        )
    }
}

fn downscale_from_lod(
    conn: DbType,
    image: Arc<Mutex<MosaicedDataset>>,
    tile_size: u64,
    lod: u64,
    multi_bar: MultiProgress,
    s: &Scope,
) {
    // let thread_pool = raycon::ThreadPoolBuilder::default().build().unwrap();

    let image_resolution = image.lock().unwrap().dataset.raster_size();

    let columns: u64 = image_resolution.0 as u64 / (tile_size * 2_u64.pow(lod as u32));
    let rows: u64 = image_resolution.1 as u64 / (tile_size * 2_u64.pow(lod as u32));

    let task_size = (columns + 1) * (rows + 1);

    let bar = ProgressBar::new(task_size);
    bar.set_message(format!("Processing lod: {}", lod));

    multi_bar.add(bar.clone());

    for i in 0..rows {
        for j in 0..columns {
            let conn = conn.clone();
            let image = image.clone();
            let bar = bar.clone();

            s.spawn(move |_s| {
                feature_extraction_to_database(
                    conn.clone(),
                    image.clone(),
                    tile_size,
                    j,
                    i,
                    lod,
                    bar,
                    task_size,
                )
            });
        }
    }
}

fn feature_extraction_to_database(
    conn: DbType,
    image: Arc<Mutex<MosaicedDataset>>,
    tile_size: u64,
    column: u64,
    row: u64,
    lod: u64,
    bar: ProgressBar,
    task_size: u64,
) {
    let tile = image
        .lock()
        .unwrap()
        .to_rgb(
            (
                (column * (tile_size * 2_u64.pow(lod as u32))) as isize,
                (row * (tile_size * 2_u64.pow(lod as u32))) as isize,
            ),
            (
                (tile_size * 2_u64.pow(lod as u32)) as usize,
                (tile_size * 2_u64.pow(lod as u32)) as usize,
            ),
            (tile_size as usize, tile_size as usize),
        )
        .expect("Could not read tile from reference image");
    let tile_mat = raster_to_mat(&tile, tile_size as i32, tile_size as i32)
        .expect("Could not convert tile to mat");

    let keypoints = akaze_keypoint_descriptor_extraction_def(&tile_mat.mat).unwrap();

    let insert_image = models::InsertImage {
        level_of_detail: &(lod as i32),
        x_start: &((column * tile_size) as i32),
        x_end: &((column * tile_size + tile_size - 1) as i32),
        y_start: &((row * tile_size) as i32),
        y_end: &((row * tile_size + tile_size - 1) as i32),
    };

    // let insert_keypoints = keypoints.into_iter().map(|keypoint| models::InsertKeypoint{
    //     x_coord: keypoint.pt
    // });

    let insert_image = imagedb::Image::One(insert_image);

    let mut conn = conn.lock().unwrap();

    let image_id = imagedb::Image::create_image(&mut conn, insert_image).unwrap();
    let db_keypoints: Vec<DbKeypoints> = keypoints
        .to_db_type(image_id)
        .into_iter()
        .map(|keypoint| DbKeypoints {
            x_coord: keypoint.x_coord + (column * tile_size * 2_u64.pow(lod as u32)) as f64,
            y_coord: keypoint.y_coord + (row * tile_size * 2_u64.pow(lod as u32)) as f64,
            ..keypoint
        })
        .collect();

    let mut insert_keypoints: Vec<models::InsertKeypoint> = Vec::with_capacity(db_keypoints.len());

    for i in 0..db_keypoints.len() {
        insert_keypoints.push(models::InsertKeypoint {
            x_coord: &db_keypoints[i].x_coord,
            y_coord: &db_keypoints[i].y_coord,
            size: &db_keypoints[i].size,
            angle: &db_keypoints[i].angle,
            response: &db_keypoints[i].response,
            octave: &db_keypoints[i].octave,
            class_id: &db_keypoints[i].class_id,
            descriptor: &db_keypoints[i].descriptor,
            image_id: &db_keypoints[i].image_id,
        });
    }

    let db_insert_keypoints = keypointdb::Keypoint::Multiple(insert_keypoints);

    keypointdb::Keypoint::create_keypoint(&mut conn, db_insert_keypoints).unwrap();

    bar.inc(1);

    if bar.length().unwrap() == tile_size {
        bar.finish();
    }
}
