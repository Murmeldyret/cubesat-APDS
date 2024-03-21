use diesel::result::Error as DieselError;
use diesel::PgConnection;

pub mod models;
pub mod schema;

pub enum Image {
    One(models::InsertImage),
    Multiple(Vec<models::InsertImage>),
}

pub trait ImageDatabase {
    fn create_image(
        conn: &mut PgConnection,
        x_start: i32,
        y_start: i32,
        x_end: i32,
        y_end: i32,
        level_of_detail: i32,
    ) -> Result<(), DieselError>;
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

