use diesel::prelude::*;

use crate::schema::*;

#[derive(Queryable, Selectable, Clone, Copy, Debug)]
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

#[derive(Insertable, Clone, Copy, Debug)]
#[diesel(table_name = ref_image)]
pub struct InsertImage<'a> {
    pub x_start: &'a i32,
    pub y_start: &'a i32,
    pub x_end: &'a i32,
    pub y_end: &'a i32,
    pub level_of_detail: &'a i32,
}

#[derive(Queryable, Selectable, Clone, Debug)]
#[diesel(table_name = keypoint)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct Keypoint {
    pub id: i32,
    pub x_coord: f64,
    pub y_coord: f64,
    pub size: f64,
    pub angle: f64,
    pub response: f64,
    pub octave: i32,
    pub class_id: i32,
    pub descriptor: Vec<u8>,
    pub image_id: i32,
}

#[derive(Insertable, Clone, Debug)]
#[diesel(table_name = keypoint)]
pub struct InsertKeypoint<'a> {
    pub x_coord: &'a f64,
    pub y_coord: &'a f64,
    pub size: &'a f64,
    pub angle: &'a f64,
    pub response: &'a f64,
    pub octave: &'a i32,
    pub class_id: &'a i32,
    pub descriptor: &'a [u8],
    pub image_id: &'a i32,
}
