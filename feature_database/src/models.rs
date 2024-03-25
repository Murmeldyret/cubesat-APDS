use diesel::prelude::*;

use crate::schema::*;

#[derive(Queryable, Selectable, Clone, Copy)]
#[diesel(table_name = ref_image)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct Image {
    pub id: i32,
    pub x_start: i32,
    pub y_start: i32,
    pub x_end: i32,
    pub y_end: i32,
    pub level_of_detail: i32,
}

#[derive(Insertable, Clone, Copy)]
#[diesel(table_name = ref_image)]
pub struct InsertImage<'a> {
    pub x_start: &'a i32,
    pub y_start: &'a i32,
    pub x_end: &'a i32,
    pub y_end: &'a i32,
    pub level_of_detail: &'a i32,
}

#[derive(Queryable, Selectable, Clone)]
#[diesel(table_name = keypoint)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct Keypoint {
    id: i32,
    x_coord: f64,
    y_coord: f64,
    size: f64,
    angle: f64,
    response: f64,
    octave: i32,
    class_id: i32,
    descriptor: Vec<u8>,
    image_id: i32,
}

#[derive(Insertable, Clone)]
#[diesel(table_name = keypoint)]
pub struct InsertKeypoint<'a> {
    x_coord: &'a f64,
    y_coord: &'a f64,
    size: &'a f64,
    angle: &'a f64,
    response: &'a f64,
    octave: &'a i32,
    class_id: &'a i32,
    descriptor: &'a Vec<u8>,
    image_id: &'a i32,
}

