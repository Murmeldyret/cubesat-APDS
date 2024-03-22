use diesel::pg::PgConnection;
use diesel::result::Error as DieselError;

pub mod models;
pub mod schema;

pub enum Image {
    One(models::InsertImage),
    Multiple(Vec<models::InsertImage>),
}

impl ImageDatabase for Image {
    fn create_image(conn: &mut PgConnection, image: Image) -> Result<(), DieselError> {
        todo!()
    }

    fn read_image_from_id(conn: &mut PgConnection, id: i32) -> Result<models::Image, DieselError> {
        todo!()
    }

    fn find_images_from_dimensions(
        conn: &mut PgConnection,
        x_start: i32,
        y_start: i32,
        x_end: i32,
        y_end: i32,
    ) -> Result<Vec<i32>, DieselError> {
        todo!()
    }

    fn find_images_from_lod(conn: &mut PgConnection, level_of_detail: i32) -> Vec<i32> {
        todo!()
    }

    fn delete_image(conn: &mut PgConnection, id: i32) -> Result<(), DieselError> {
        todo!()
    }
}

pub trait ImageDatabase {
    fn create_image(conn: &mut PgConnection, image: Image) -> Result<(), DieselError>;
    fn read_image_from_id(conn: &mut PgConnection, id: i32) -> Result<models::Image, DieselError>;
    fn find_images_from_dimensions(
        conn: &mut PgConnection,
        x_start: i32,
        y_start: i32,
        x_end: i32,
        y_end: i32,
    ) -> Result<Vec<i32>, DieselError>;
    fn find_images_from_lod(conn: &mut PgConnection, level_of_detail: i32) -> Vec<i32>;
    fn delete_image(conn: &mut PgConnection, id: i32) -> Result<(), DieselError>;
}

pub enum Keypoint {
    One(models::InsertKeypoint),
    Multiple(Vec<models::InsertKeypoint>),
}

pub trait KeypointDatabase {
    fn create_keypoint(conn: &mut PgConnection, keypoint: Keypoint) -> Result<(), DieselError>;
    fn read_keypoint_from_id(
        conn: &mut PgConnection,
        id: i32,
    ) -> Result<models::Keypoint, DieselError>;
    fn read_keypoints_from_image_id(
        conn: &mut PgConnection,
        image_id: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError>;
    fn read_keypoints_from_lod(
        conn: &mut PgConnection,
        level: u32,
    ) -> Result<models::Keypoint, DieselError>;
    fn read_keypoints_from_coordinates(
        conn: &mut PgConnection,
        x_start: i32,
        y_start: i32,
        x_end: i32,
        y_end: i32,
    ) -> Result<models::Keypoint, DieselError>;
    fn delete_keypoint(conn: &mut PgConnection, id: i32) -> Result<(), DieselError>;
}

pub enum Descriptor {
    One(models::InsertDescriptor),
    Multiple(Vec<models::InsertDescriptor>),
}

pub trait DescriptorDatabase {
    fn create_descriptor(
        conn: &mut PgConnection,
        descriptor: Descriptor,
    ) -> Result<(), DieselError>;
    fn read_discriptor_from_id(
        conn: &mut PgConnection,
        id: i32,
    ) -> Result<models::Descriptor, DieselError>;
    fn read_discriptor_from_ids(
        conn: &mut PgConnection,
        ids: &[i32],
    ) -> Result<Vec<models::Descriptor>, DieselError>;
    fn delete_descriptor(conn: &mut PgConnection, id: i32) -> Result<(), DieselError>;
}

#[cfg(test)]
mod image_tests {
    use std::env;

    use self::schema::image::dsl::image;
    use super::*;
    use diesel::prelude::*;
    use dotenvy::dotenv;

    fn setup_test_database() -> PgConnection {
        dotenv().ok();

        let database_url =
            env::var("DATABASE_URL").expect("DATABASE_URL must be set for tests to work");

        Connection::establish(&database_url)
            .unwrap_or_else(|_| panic!("Error connecting to {}", database_url))
    }

    #[test]
    fn image_creation() {
        let connection = &mut setup_test_database();

        let insert_image = models::InsertImage {
            x_start: 0,
            y_start: 0,
            x_end: 10,
            y_end: 10,
            level_of_detail: 1,
        };

        let inserted_image = Image::One(insert_image);

        Image::create_image(&mut connection, inserted_image)
            .expect("Could not add image to database");

        let image = image
            .find(1)
            .select(models::Image::as_select())
            .first(&mut connection);
    }
}

