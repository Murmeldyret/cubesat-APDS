pub mod imagedb;
pub mod keypointdb;
pub mod models;
pub mod schema;

pub mod db_helpers {
    use std::env;
    use std::sync::{Arc, Mutex};

    use diesel::prelude::*;
    use dotenvy::dotenv;
    use once_cell::sync::Lazy;

    static DATABASE_LOCK: Lazy<Arc<Mutex<i32>>> = Lazy::new(|| Arc::new(Mutex::new(0)));
    static RESERVER_LOCK: Lazy<Arc<Mutex<i32>>> = Lazy::new(|| Arc::new(Mutex::new(0)));

    pub fn obtain_lock() -> std::sync::MutexGuard<'static, i32> {
        let _lock = RESERVER_LOCK.lock().unwrap();

        let lock = DATABASE_LOCK.lock();

        if let Err(poison_lock) = lock {
            return poison_lock.into_inner();
        }

        lock.unwrap()
    }

    pub fn setup_database() -> PgConnection {
        dotenv().ok();

        let database_url =
            env::var("DATABASE_URL").expect("DATABASE_URL must be set for tests to work");

        let mut connection = Connection::establish(&database_url)
            .unwrap_or_else(|_| panic!("Error connecting to {}", database_url));

        // TODO: This can be done smarter :)
        diesel::sql_query("DELETE FROM keypoint")
            .execute(&mut connection)
            .unwrap();
        diesel::sql_query("ALTER SEQUENCE keypoint_id_seq RESTART WITH 1")
            .execute(&mut connection)
            .unwrap();
        diesel::sql_query("DELETE FROM ref_image")
            .execute(&mut connection)
            .unwrap();
        diesel::sql_query("ALTER SEQUENCE ref_image_id_seq RESTART WITH 1")
            .execute(&mut connection)
            .unwrap();

        connection
    }
}
