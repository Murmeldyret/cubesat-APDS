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
pub struct Args {
    /// The path to the folder where datasets are stored on disk
    #[command(subcommand)]
    dataset_path: DatasetPath,

    /// The path to the temp folder which shall contain processed files (/tmp will be used if not provided)
    #[arg(long)]
    temp_path: Option<String>,

    /// The database url to connect to. Can also be provided by setting environment variable: DATABASE_URL
    #[arg(long)]
    database_url: Option<String>,

    /// The number of CPU threads to use for processing arg
    #[arg(short, long, default_value_t = 1)]
    cpu_num: usize,

    /// Calculate the amount of levels of detail
    #[arg(long)]
    calculate_lod: bool,

    /// The amount of levels of details that the reference image is going to be split into
    #[arg(short, long, default_value_t = 1)]
    lod: u64,

    /// The path to the optional elevation dataset
    #[arg(short, long)]
    elevation_path: Option<String>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum DatasetPath {
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

    if args.calculate_lod == true {
        level_of_detail::calculate_level_of_detail_resolution(&args);
    }

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
    let temp_dir = tempdir().expect(
        // Creates a tmp directory which stores the dataset if no path is provided.
        "Could not create temp directory\nPlease provide directory in the parameters",
    );

    let temp_string = temp_dir.path().to_string_lossy().to_string();

    let temp_string = args.temp_path.as_ref().unwrap_or(&temp_string);

    // Not pretty, but it works.
    let mosaic = match args.dataset_path {
        DatasetPath::Dataset { path } => read_dataset(Some(path), None, &temp_string).unwrap(),
        DatasetPath::Mosaic { path } => read_dataset(None, Some(path), &temp_string).unwrap(),
    };

    if args.elevation_path.is_some() {
        mosaic.lock().unwrap().set_elevation_dataset(&args.elevation_path.expect("Elevation dataset path not found"), &temp_string).expect("Could not add elevation data to dataset");
    }

    if mosaic.lock().unwrap().elevation.is_some() {
        add_elevation(db_connection.clone(), mosaic.clone());
    }

    thread_pool.scope(move |s| {
        // Scope prevents the main process from quiting before all threads are done.
        println!("Processing mosaic");

        process_lod_from_mosaic(db_connection, mosaic, args.lod, s);
    });


    temp_dir.close().expect("Failed to delete temporary data");
}


/// This function is only called when the elevation dataset is known to exist.
fn add_elevation(conn: DbType, mosaic: Arc<Mutex<MosaicedDataset>>) {
    use feature_database::elevationdb::{geotransform, elevation};
    let mosaic = mosaic.lock().unwrap();
    let conn = &mut conn.lock().unwrap();

    let dataset_trans = mosaic.dataset.geo_transform().expect("Could not get geotransform from dataset");
    let elevation_trans = mosaic.elevation.as_ref().unwrap().geo_transform().expect("Could not get geotransform from elevation");

    geotransform::create_geotransform(conn, "dataset", dataset_trans).expect("Could not add dataset geotransform to database");
    geotransform::create_geotransform(conn, "elevation", elevation_trans).expect("Could not add dataset geotransform to database");

    elevation::add_elevation_data(conn, &mosaic.elevation.as_ref().expect("Elevation data not found")).expect("Elevation data could not be added to database");
}

fn read_dataset(dataset_path: Option<String>, mosaic_path: Option<String>, temp_string: &str) -> Result<Arc<Mutex<MosaicedDataset>>, std::io::Error> {
    // let mosaic: Arc<Mutex<MosaicedDataset>>;

    if let Some(path) = mosaic_path {
        return Ok(Arc::new(Mutex::new(MosaicedDataset::import_mosaic_dataset(&path).expect("Could not open dataset"))));
    }

    if let Some(path) = dataset_path {
            let dataset = image_extractor::RawDataset::import_datasets(&path)
                .expect("Could not open datasets");
            println!("Converting dataset to mosaic");
            return Ok(Arc::new(Mutex::new(
                dataset
                    .to_mosaic_dataset(&temp_string)
                    .expect("Could not convert dataset."),
            )));
    }

    Err(std::io::Error::new(std::io::ErrorKind::NotFound, "Could not read dataset"))
}

// A function that initialize the downscaling and extraction of each level of detail.
fn process_lod_from_mosaic(
    conn: DbType,
    image: Arc<Mutex<MosaicedDataset>>,
    lod: u64,
    s: &Scope,
) {
    let image_resolution = image
        .lock()
        .unwrap()
        .get_dimensions()
        .expect("Could not read image resolution");

    println!("Amount of lod: {}", &lod);

    let multi_bar = MultiProgress::new();
    multi_bar
        .println(format!("Processing {} level of detail", lod))
        .unwrap();

    // Loop that initiate the process for all levels of detail.
    for i in 0..lod {
        downscale_from_lod(
            conn.clone(),
            image.clone(),
            lod,
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
    amount_lod: u64,
    lod: u64,
    multi_bar: MultiProgress,
    s: &Scope,
) {
    // let thread_pool = raycon::ThreadPoolBuilder::default().build().unwrap();

    let image_resolution = image.lock().unwrap().dataset.raster_size();

    dbg!(&image_resolution);
    dbg!(&lod);

    let tile_size = (image_resolution.0 as u64/ 2_u64.pow(amount_lod as u32 - 1), image_resolution.1 as u64 / 2_u64.pow(amount_lod as u32 - 1));

    // Calculate how many columns and rows there are in the level.
    let columns: u64 = image_resolution.0 as u64 / (tile_size.0 * 2_u64.pow(lod as u32));
    let rows: u64 = image_resolution.1 as u64 / (tile_size.1 * 2_u64.pow(lod as u32));

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
    tile_size: (u64, u64),
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
                (column * (tile_size.0 * 2_u64.pow(lod as u32))) as isize,
                (row * (tile_size.1 * 2_u64.pow(lod as u32))) as isize,
            ),
            (
                (tile_size.0 * 2_u64.pow(lod as u32)) as usize,
                (tile_size.1 * 2_u64.pow(lod as u32)) as usize,
            ),
            (tile_size.0 as usize, tile_size.1 as usize),
        )
        .expect("Could not read tile from reference image");
    // Convert the tile to an openCV mat
    let tile_mat = raster_to_mat(&tile, tile_size.0 as i32, tile_size.1 as i32)
        .expect("Could not convert tile to mat");
    // Extract keypoints and descriptors
    let keypoints = akaze_keypoint_descriptor_extraction_def(&tile_mat.mat, None).unwrap();

    // Insert the image into the database.
    let insert_image = models::InsertImage {
        level_of_detail: &(lod as i32),
        x_start: &((column * tile_size.0 * 2_u64.pow(lod as u32)) as i32),
        x_end: &((column * tile_size.0 * 2_u64.pow(lod as u32) + tile_size.0 * 2_u64.pow(lod as u32) - 1) as i32),
        y_start: &((row * tile_size.1 * 2_u64.pow(lod as u32)) as i32),
        y_end: &((row * tile_size.1 * 2_u64.pow(lod as u32) + tile_size.1 * 2_u64.pow(lod as u32) - 1) as i32),
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
            x_coord: keypoint.x_coord * 2_f32.powi(lod as i32) + (column * tile_size.0 * 2_u64.pow(lod as u32)) as f32,
            y_coord: keypoint.y_coord * 2_f32.powi(lod as i32) + (row * tile_size.1 * 2_u64.pow(lod as u32)) as f32,
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


