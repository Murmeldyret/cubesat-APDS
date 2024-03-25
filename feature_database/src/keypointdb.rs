use diesel::pg::PgConnection;
use diesel::result::Error as DieselError;
use crate::models;

pub enum Keypoint<'a> {
    One(models::InsertKeypoint<'a>),
    Multiple(Vec<models::InsertKeypoint<'a>>),
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

