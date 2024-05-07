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
    pub x_coord: f32,
    pub y_coord: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
    pub class_id: i32,
    pub descriptor: Vec<u8>,
    pub image_id: i32,
}

#[derive(Insertable, Clone, Debug)]
#[diesel(table_name = keypoint)]
pub struct InsertKeypoint<'a> {
    pub x_coord: &'a f32,
    pub y_coord: &'a f32,
    pub size: &'a f32,
    pub angle: &'a f32,
    pub response: &'a f32,
    pub octave: &'a i32,
    pub class_id: &'a i32,
    pub descriptor: &'a [u8],
    pub image_id: &'a i32,
}

#[derive(Queryable, Selectable, Clone, Debug)]
#[diesel(table_name = geotransform)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct GeoTransform {
    pub id: i32,
    pub dataset_name: String,
    pub transform: Vec<Option<f64>>,
}

#[derive(Insertable, Clone, Debug)]
#[diesel(table_name = geotransform)]
pub struct InsertGeoTransform<'a> {
    pub dataset_name: &'a str,
    pub transform: &'a [f64],
}

#[derive(Queryable, Selectable, Clone, Debug)]
#[diesel(table_name = elevation)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct Elevation {
    pub id: i32,
    pub height: f64,
}

#[derive(Insertable, Clone, Debug)]
#[diesel(table_name = elevation)]
pub struct InsertElevation<'a> {
    pub height: &'a f64,
}

#[derive(Queryable, Selectable, Clone, Debug)]
#[diesel(table_name = elevation_properties)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct ElevationProperties {
    pub id: i32,
    pub x_size: i32,
    pub y_size: i32,
}

#[derive(Insertable, Clone, Debug)]
#[diesel(table_name = elevation_properties)]
pub struct InsertElevationProperties<'a> {
    pub x_size: &'a i32,
    pub y_size: &'a i32,
}
