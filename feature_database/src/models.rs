use diesel::{deserialize::Queryable, insertable::Insertable, Selectable};

use diesel::prelude::*;

use crate::schema::*;

#[derive(Queryable, Selectable, Insertable)]
#[diesel(table_name = level_of_detail)]
pub struct LevelOfDetail {
    id: i32,
}

#[derive(Queryable, Selectable, Insertable)]
#[diesel(table_name = image)]
pub struct Image {
    id: i32,
    x_start: i32,
    y_start: i32,
    x_end: i32,
    y_end: i32,
    level_of_detail: i32,
}

#[derive(Queryable, Selectable, Insertable)]
#[diesel(table_name = keypoint)]
pub struct Keypoint {
    id: i32,
    x_coord: f32,
    y_coord: f32,
    size: f32,
    angle: f32,
    response: f32,
    octave: i32,
    class_id: i32,
    image_id: i32,
}

#[derive(Queryable, Selectable, Insertable)]
#[diesel(table_name = descriptor)]
pub struct Descriptor {
    id: i32,
    value: Vec<u8>,
}

