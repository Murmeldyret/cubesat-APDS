use diesel::PgConnection;
use dotenvy::dotenv;
use feature_database::{
    imagedb, imagedb::ImageDatabase, keypointdb, keypointdb::KeypointDatabase, models,
};
use feature_extraction::{akaze_keypoint_descriptor_extraction_def, DbKeypoints};
use geotiff_lib::image_extractor;
use geotiff_lib::image_extractor::{Datasets, MosaicDataset, MosaicedDataset};
use homographier::homographier::raster_to_mat;
use indicatif::{MultiProgress, ProgressBar};
use tempfile::tempdir;

use level_of_detail::calculate_amount_of_levels;
use raycon::Scope;
use rayon as raycon;

use clap::{Parser, Subcommand};
use std::sync::{Arc, Mutex};

pub mod level_of_detail;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path to the folder where datasets are stored on disk
    #[command(subcommand)]
    dataset_path: DatasetPath,

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

#[derive(Subcommand, Debug, Clone)]
enum DatasetPath {
    /// Load a raw dataset from disk
    Dataset {
        /// The path to the raw dataset
        path: String,
    },
    /// Load a preprocced mosaic from disk
    Mosaic {
        /// The path to the mosaiced dataset
        path: String,
    },
}

type DbType = Arc<Mutex<PgConnection>>;

fn main() {
    dotenv().expect("Could not read .env file");

    let args = Args::parse();

    // Must be in mutex since diesel is a sync library.
    let db_connection: DbType =
        Arc::new(Mutex::new(feature_database::db_helpers::setup_database()));

    println!("Read dataset");

    // Create a threadpool for multithreading.
    let thread_pool = raycon::ThreadPoolBuilder::new()
        .num_threads(args.cpu_num)
        .build()
        .expect("Could not create thread pool");

    // A GDAL Dataset is not threadsafe. Therefore Arc<Mutex<_>> is necessary.
    let mosaic: Arc<Mutex<MosaicedDataset>>;
    let temp_dir = tempdir().expect(
        // Creates a tmp directory which stores the dataset if no path is provided.
        "Could not create temp directory\nPlease provide directory in the parameters",
    );

    let temp_string = temp_dir.path().to_string_lossy().to_string();

    match args.dataset_path {
    DatasetPath::Dataset { path } => {
        let temp_path = args.temp_path.unwrap_or(temp_string);
        let dataset = image_extractor::RawDataset::import_datasets(
            &path,
        )
        .expect("Could not open datasets");
        println!("Converting dataset to mosaic");
        mosaic = Arc::new(Mutex::new(
            dataset
                .to_mosaic_dataset(&temp_path)
                .expect("Could not convert dataset."),
        ));
    },
    DatasetPath::Mosaic { path } => {
        mosaic = Arc::new(Mutex::new(
            MosaicedDataset::import_mosaic_dataset(
                &path,
            )
            .expect("Could not read mosaic"),
        ));
    },}

    thread_pool.scope(move |s| {
        // Scope prevents the main process from quiting before all threads are done.
        println!("Processing mosaic");

        process_lod_from_mosaic(db_connection, mosaic, args.tile_size, s);
    });
    temp_dir.close().unwrap()
}

/// A function that initialize the downscaling and extraction of each level of detail.
fn process_lod_from_mosaic(
    conn: DbType,
    image: Arc<Mutex<MosaicedDataset>>,
    tile_size: u64,
    s: &Scope,
) {
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

    // Loop that initiate the process for all levels of detail.
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

/// A function that downscale and process a specific level of detail.
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

    // Calculate how many columns and rows there are in the level.
    let columns: u64 = image_resolution.0 as u64 / (tile_size * 2_u64.pow(lod as u32));
    let rows: u64 = image_resolution.1 as u64 / (tile_size * 2_u64.pow(lod as u32));

    // Computes the amount of tasks that has to be done.
    let task_size = columns * rows;

    let bar = ProgressBar::new(task_size);
    bar.set_message(format!("Processing lod: {}", lod));

    multi_bar.add(bar.clone());

    // The loop that spawns a thread to proccess keypoints for every tile. Tiles are queued by the threadpool.
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
) {
    // Read a tile from the dataset.
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
    // Convert the tile to an openCV mat
    let tile_mat = raster_to_mat(&tile, tile_size as i32, tile_size as i32)
        .expect("Could not convert tile to mat");
    // Extract keypoints and descriptors
    let keypoints = akaze_keypoint_descriptor_extraction_def(&tile_mat.mat).unwrap();

    // Insert the image into the database.
    let insert_image = models::InsertImage {
        level_of_detail: &(lod as i32),
        x_start: &((column * tile_size) as i32),
        x_end: &((column * tile_size + tile_size - 1) as i32),
        y_start: &((row * tile_size) as i32),
        y_end: &((row * tile_size + tile_size - 1) as i32),
    };

    let insert_image = imagedb::Image::One(insert_image);

    // let mut conn = conn.lock().unwrap();

    // Insert image into database.
    let image_id = imagedb::Image::create_image(&mut conn.lock().unwrap(), insert_image).unwrap();

    // Convert keypoints to db_keypoints.
    let db_keypoints: Vec<DbKeypoints> = keypoints
        .to_db_type(image_id)
        .into_iter()
        .map(|keypoint| DbKeypoints {
            x_coord: keypoint.x_coord + (column * tile_size * 2_u64.pow(lod as u32)) as f32,
            y_coord: keypoint.y_coord + (row * tile_size * 2_u64.pow(lod as u32)) as f32,
            ..keypoint
        })
        .collect();

    let mut insert_keypoints: Vec<models::InsertKeypoint> = Vec::with_capacity(db_keypoints.len());
    // Make keypoints a reference because diesel gets sad when it owns data.
    for keypoint in &db_keypoints {
        insert_keypoints.push(models::InsertKeypoint {
            x_coord: &keypoint.x_coord,
            y_coord: &keypoint.y_coord,
            size: &keypoint.size,
            angle: &keypoint.angle,
            response: &keypoint.response,
            octave: &keypoint.octave,
            class_id: &keypoint.class_id,
            descriptor: &keypoint.descriptor,
            image_id: &keypoint.image_id,
        });
    }

    let db_insert_keypoints = keypointdb::Keypoint::Multiple(insert_keypoints);

    keypointdb::Keypoint::create_keypoint(&mut conn.lock().unwrap(), db_insert_keypoints).unwrap();

    bar.inc(1);
}
