use diesel::prelude::*;

use crate::schema::*;

#[derive(Queryable, Selectable)]
#[diesel(table_name = image)]
pub struct Image {
    pub id: i32,
    pub x_start: i32,
    pub y_start: i32,
    pub x_end: i32,
    pub y_end: i32,
    pub level_of_detail: i32,
}

#[derive(Insertable)]
#[diesel(table_name = image)]
pub struct InsertImage {
    pub x_start: i32,
    pub y_start: i32,
    pub x_end: i32,
    pub y_end: i32,
    pub level_of_detail: i32,
}

#[derive(Queryable, Selectable)]
#[diesel(table_name = keypoint)]
pub struct Keypoint {
    id: i32,
    x_coord: f64,
    y_coord: f64,
    size: f64,
    angle: f64,
    response: f64,
    octave: i32,
    class_id: i32,
    image_id: i32,
}

#[derive(Insertable)]
#[diesel(table_name = keypoint)]
pub struct InsertKeypoint {
    x_coord: f64,
    y_coord: f64,
    size: f64,
    angle: f64,
    response: f64,
    octave: i32,
    class_id: i32,
    image_id: i32,
}

#[derive(Queryable, Selectable)]
#[diesel(table_name = descriptor)]
pub struct Descriptor {
    id: i32,
    value: Vec<u8>,
}

#[derive(Queryable, Selectable)]
#[diesel(table_name = descriptor)]
pub struct InsertDescriptor {
    id: i32,
    value: Vec<u8>,
}

