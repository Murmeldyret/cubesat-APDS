use crate::schema::ref_image::dsl;
use diesel::pg::PgConnection;
use diesel::prelude::*;
use diesel::result::Error as DieselError;

use crate::models;

pub enum Image<'a> {
    One(models::InsertImage<'a>),
    Multiple(Vec<models::InsertImage<'a>>),
}

impl ImageDatabase for Image<'_> {
    fn create_image(conn: &mut PgConnection, input_image: Image) -> Result<i32, DieselError> {
        match input_image {
            Image::One(single_image) => return Ok(create_image_in_database(conn, &single_image)?),
            Image::Multiple(multiple_images) => {
                let result: Result<Vec<i32>, DieselError> = multiple_images
                    .into_iter()
                    .map(|img| create_image_in_database(conn, &img))
                    .collect();

                match result {
                    Ok(image_vec) => return Ok(image_vec[0]),
                    Err(e) => return Err(e),
                }
            }
        }
    }

    fn read_image_from_id(conn: &mut PgConnection, id: i32) -> Result<models::Image, DieselError> {
        dsl::ref_image
            .find(id)
            .select(models::Image::as_select())
            .first(conn)
    }

    fn find_images_from_dimensions(
        conn: &mut PgConnection,
        x_start: i32,
        y_start: i32,
        x_end: i32,
        y_end: i32,
        level_of_detail: i32,
    ) -> Result<Vec<i32>, DieselError> {
        dsl::ref_image
            .filter(dsl::x_end.ge(x_start))
            .filter(dsl::x_start.le(x_end))
            .filter(dsl::y_end.ge(y_start))
            .filter(dsl::y_start.le(y_end))
            .filter(dsl::level_of_detail.eq(level_of_detail))
            .select(dsl::id)
            .load(conn)
    }

    fn find_images_from_lod(
        conn: &mut PgConnection,
        level_of_detail: i32,
    ) -> Result<Vec<i32>, DieselError> {
        dsl::ref_image
            .filter(dsl::level_of_detail.eq(level_of_detail))
            .select(dsl::id)
            .load(conn)
    }

    fn delete_image(conn: &mut PgConnection, id: i32) -> Result<(), DieselError> {
        match diesel::delete(dsl::ref_image.find(id)).execute(conn) {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }
}

fn create_image_in_database(
    connection: &mut PgConnection,
    insert_image: &models::InsertImage,
) -> Result<i32, DieselError> {
    let result: Result<models::Image, DieselError> = diesel::insert_into(crate::schema::ref_image::table)
        .values(insert_image)
        .returning(models::Image::as_returning())
        .get_result(connection);

    match result {
        Ok(image_db) => Ok(image_db.id),
        Err(e) => Err(e),
    }
}

pub trait ImageDatabase {
    fn create_image(conn: &mut PgConnection, image: Image) -> Result<i32, DieselError>;
    fn read_image_from_id(conn: &mut PgConnection, id: i32) -> Result<models::Image, DieselError>;
    fn find_images_from_dimensions(
        conn: &mut PgConnection,
        x_start: i32,
        y_start: i32,
        x_end: i32,
        y_end: i32,
        level_of_detail: i32,
    ) -> Result<Vec<i32>, DieselError>;
    fn find_images_from_lod(
        conn: &mut PgConnection,
        level_of_detail: i32,
    ) -> Result<Vec<i32>, DieselError>;
    fn delete_image(conn: &mut PgConnection, id: i32) -> Result<(), DieselError>;
}

#[cfg(test)]
mod image_tests {
    use super::*;
    use crate::db_helpers::{obtain_lock, setup_database};
    use crate::schema::ref_image::dsl::*;

    #[test]
    fn image_creation() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        let insert_image = models::InsertImage {
            x_start: &0,
            y_start: &0,
            x_end: &10,
            y_end: &10,
            level_of_detail: &1,
        };

        let inserted_image = Image::One(insert_image);

        Image::create_image(connection, inserted_image).expect("Could not add image to database");

        let fetched_image: models::Image = ref_image
            .find(1)
            .select(models::Image::as_select())
            .first(connection)
            .expect("Could not find created image");

        assert_eq!(fetched_image.x_start, *insert_image.x_start);
        assert_eq!(fetched_image.y_start, *insert_image.y_start);
        assert_eq!(fetched_image.x_end, *insert_image.x_end);
        assert_eq!(fetched_image.y_end, *insert_image.y_end);
        assert_eq!(fetched_image.level_of_detail, *insert_image.level_of_detail);
    }

    #[test]
    fn image_fetching_id() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        let insert_image = models::InsertImage {
            x_start: &0,
            y_start: &0,
            x_end: &10,
            y_end: &10,
            level_of_detail: &1,
        };

        diesel::insert_into(crate::schema::ref_image::table)
            .values(&insert_image)
            .returning(models::Image::as_returning())
            .get_result(connection)
            .expect("Error saving new image");

        let fetched_image =
            Image::read_image_from_id(connection, 1).expect("Could not read image from database");

        assert_eq!(fetched_image.x_start, *insert_image.x_start);
        assert_eq!(fetched_image.y_start, *insert_image.y_start);
        assert_eq!(fetched_image.x_end, *insert_image.x_end);
        assert_eq!(fetched_image.y_end, *insert_image.y_end);
        assert_eq!(fetched_image.level_of_detail, *insert_image.level_of_detail);
    }

    #[test]
    fn image_fetching_id_not_available() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        let fetched_image = Image::read_image_from_id(connection, 1);

        assert!(fetched_image.is_err_and(|e| e.eq(&DieselError::NotFound)));
    }

    #[test]
    fn image_fetching_dimensions() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        // TODO: Make a generator of images
        let insert_images = vec![
            models::InsertImage {
                x_start: &0,
                y_start: &0,
                x_end: &9,
                y_end: &9,
                level_of_detail: &1,
            },
            models::InsertImage {
                x_start: &10,
                y_start: &0,
                x_end: &19,
                y_end: &9,
                level_of_detail: &1,
            },
            models::InsertImage {
                x_start: &0,
                y_start: &10,
                x_end: &9,
                y_end: &19,
                level_of_detail: &1,
            },
            models::InsertImage {
                x_start: &10,
                y_start: &10,
                x_end: &19,
                y_end: &19,
                level_of_detail: &1,
            },
        ];

        insert_images.into_iter().for_each(|single_image| {
            diesel::insert_into(crate::schema::ref_image::table)
                .values(&single_image)
                .returning(models::Image::as_returning())
                .get_result(connection)
                .expect("Error saving new image");
        });

        let fetched_image_ids = Image::find_images_from_dimensions(connection, 3, 0, 15, 7, 1);

        assert!(fetched_image_ids.is_ok_and(|ids| ids.contains(&1) && ids.contains(&2)));
    }

    #[test]
    fn images_fetched_from_lod() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        // TODO: Make a generator of images
        let insert_images = vec![
            models::InsertImage {
                x_start: &0,
                y_start: &0,
                x_end: &9,
                y_end: &9,
                level_of_detail: &1,
            },
            models::InsertImage {
                x_start: &10,
                y_start: &0,
                x_end: &19,
                y_end: &9,
                level_of_detail: &1,
            },
            models::InsertImage {
                x_start: &0,
                y_start: &10,
                x_end: &9,
                y_end: &19,
                level_of_detail: &1,
            },
            models::InsertImage {
                x_start: &10,
                y_start: &10,
                x_end: &19,
                y_end: &19,
                level_of_detail: &1,
            },
        ];

        insert_images.into_iter().for_each(|single_image| {
            diesel::insert_into(crate::schema::ref_image::table)
                .values(&single_image)
                .returning(models::Image::as_returning())
                .get_result(connection)
                .expect("Error saving new image");
        });

        let image_ids = Image::find_images_from_lod(connection, 1);

        assert!(image_ids.is_ok_and(|ids| (1..=4)
            .into_iter()
            .fold(false, |acc, i| acc || ids.contains(&i))));
    }

    #[test]
    fn image_deletion() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        let insert_images = vec![
            models::InsertImage {
                x_start: &0,
                y_start: &0,
                x_end: &9,
                y_end: &9,
                level_of_detail: &1,
            },
            models::InsertImage {
                x_start: &10,
                y_start: &0,
                x_end: &19,
                y_end: &9,
                level_of_detail: &1,
            },
            models::InsertImage {
                x_start: &0,
                y_start: &10,
                x_end: &9,
                y_end: &19,
                level_of_detail: &1,
            },
            models::InsertImage {
                x_start: &10,
                y_start: &10,
                x_end: &19,
                y_end: &19,
                level_of_detail: &1,
            },
        ];

        insert_images.into_iter().for_each(|single_image| {
            diesel::insert_into(crate::schema::ref_image::table)
                .values(&single_image)
                .returning(models::Image::as_returning())
                .get_result(connection)
                .expect("Error saving new image");
        });

        let result = Image::delete_image(connection, 1);

        let db_result = ref_image
            .select(models::Image::as_select())
            .load(connection)
            .expect("Error loading images");

        assert!(result.is_ok());
        assert_eq!(db_result.len(), 3);
    }
}
