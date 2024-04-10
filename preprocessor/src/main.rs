use feature_database;
use feature_extraction;
use geotiff_lib;
use dotenvy::dotenv;

pub mod level_of_detail;

fn main() {
    dotenv().expect("Could not read .env file");

    let db_connection = &mut feature_database::db_helpers::setup_database();


}

