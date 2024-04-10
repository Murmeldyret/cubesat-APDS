use crate::models;
use crate::schema::keypoint::dsl;
use diesel::pg::PgConnection;
use diesel::prelude::*;
use diesel::result::Error as DieselError;

pub enum Keypoint<'a> {
    One(models::InsertKeypoint<'a>),
    Multiple(Vec<models::InsertKeypoint<'a>>),
}

impl<'a> KeypointDatabase for Keypoint<'a> {
    fn create_keypoint(
        conn: &mut PgConnection,
        input_keypoint: Keypoint,
    ) -> Result<(), DieselError> {
        match input_keypoint {
            Keypoint::One(single_image) => create_keypoint_in_database(conn, &single_image)?,
            Keypoint::Multiple(multiple_images) => {
                let result: Result<Vec<()>, DieselError> = multiple_images
                    .into_iter()
                    .map(|key| create_keypoint_in_database(conn, &key))
                    .collect();
                match result {
                    Ok(_) => return Ok(()),
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(())
    }

    fn read_keypoint_from_id(
        conn: &mut PgConnection,
        id: i32,
    ) -> Result<models::Keypoint, DieselError> {
        dsl::keypoint
            .find(id)
            .select(models::Keypoint::as_select())
            .first(conn)
    }

    fn read_keypoints_from_image_id(
        conn: &mut PgConnection,
        image_id: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError> {
        dsl::keypoint
            .filter(dsl::image_id.eq(image_id))
            .select(models::Keypoint::as_select())
            .load(conn)
    }

    fn read_keypoints_from_lod(
        conn: &mut PgConnection,
        level_of_detail: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError> {
        use crate::schema::ref_image;
        let image_ids: Vec<i32> = ref_image::dsl::ref_image
            .filter(ref_image::dsl::level_of_detail.eq(level_of_detail))
            .select(ref_image::dsl::id)
            .load(conn)?;

        let keypoints_vec: Vec<Vec<models::Keypoint>> = image_ids
            .into_iter()
            .flat_map(|id| {
                dsl::keypoint
                    .filter(dsl::image_id.eq(id))
                    .select(models::Keypoint::as_select())
                    .load(conn)
            })
            .collect();

        let keypoints_vec = keypoints_vec.into_iter().flatten().collect();

        Ok(keypoints_vec)
    }

    fn read_keypoints_from_coordinates(
        conn: &mut PgConnection,
        x_start: f64,
        y_start: f64,
        x_end: f64,
        y_end: f64,
        level_of_detail: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError> {
        use crate::imagedb::ImageDatabase;
        let image_ids = crate::imagedb::Image::find_images_from_dimensions(
            conn,
            x_start.floor() as i32,
            y_start.floor() as i32,
            x_end.ceil() as i32,
            y_end.ceil() as i32,
            level_of_detail,
        )?;

        let keypoints_vec: Result<Vec<Vec<models::Keypoint>>, DieselError> = image_ids
            .into_iter()
            .map(|id| {
                dsl::keypoint
                    .filter(dsl::image_id.eq(id))
                    .filter(dsl::x_coord.ge(x_start.floor()))
                    .filter(dsl::x_coord.le(x_end.ceil()))
                    .filter(dsl::y_coord.ge(y_start.floor()))
                    .filter(dsl::y_coord.le(y_end.ceil()))
                    .select(models::Keypoint::as_select())
                    .load(conn)
            })
            .collect();

        let keypoints_vec: Vec<models::Keypoint> = keypoints_vec?.into_iter().flatten().collect();

        Ok(keypoints_vec)
    }

    fn delete_keypoint(conn: &mut PgConnection, id: i32) -> Result<(), DieselError> {
        match diesel::delete(dsl::keypoint.find(id)).execute(conn) {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }
}

fn create_keypoint_in_database(
    connection: &mut PgConnection,
    input_keypoint: &models::InsertKeypoint,
) -> Result<(), DieselError> {
    let result = diesel::insert_into(crate::schema::keypoint::table)
        .values(input_keypoint)
        .returning(models::Keypoint::as_returning())
        .get_result(connection);

    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(e),
    }
}

pub trait KeypointDatabase {
    fn create_keypoint(
        conn: &mut PgConnection,
        input_keypoint: Keypoint,
    ) -> Result<(), DieselError>;
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
        level_of_detail: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError>;
    fn read_keypoints_from_coordinates(
        conn: &mut PgConnection,
        x_start: f64,
        y_start: f64,
        x_end: f64,
        y_end: f64,
        level_of_detail: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError>;
    fn delete_keypoint(conn: &mut PgConnection, id: i32) -> Result<(), DieselError>;
}

#[cfg(test)]
mod tests {
    use self::models::InsertImage;

    use super::*;
    use crate::schema::keypoint::dsl::*;
    use crate::db_helpers::{obtain_lock, setup_database};

    fn generate_images_in_database(connection: &mut PgConnection, amount: i32) {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        for _i in 0..amount {
            let insert_image = InsertImage {
                x_start: &rng.gen(),
                y_start: &rng.gen(),
                x_end: &rng.gen(),
                y_end: &rng.gen(),
                level_of_detail: &rng.gen(),
            };

            diesel::insert_into(crate::schema::ref_image::table)
                .values(insert_image)
                .returning(models::Image::as_returning())
                .get_result(connection)
                .expect("Could not generate images to the database");
        }
    }

    #[test]
    fn keypoint_creation() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        generate_images_in_database(connection, 1);

        let insert_keypoint = models::InsertKeypoint {
            x_coord: &1.0,
            y_coord: &1.5,
            size: &2.0,
            angle: &2.5,
            response: &3.0,
            octave: &4,
            class_id: &5,
            descriptor: &[6_u8],
            image_id: &1,
        };

        let keypoint_enum = Keypoint::One(insert_keypoint.clone());

        Keypoint::create_keypoint(connection, keypoint_enum)
            .expect("Could not add keypoint to database");

        let fetched_keypoint: models::Keypoint = keypoint
            .find(1)
            .select(models::Keypoint::as_select())
            .first(connection)
            .expect("Could not find created keypoint");

        assert_eq!(fetched_keypoint.x_coord, *insert_keypoint.x_coord);
        assert_eq!(fetched_keypoint.y_coord, *insert_keypoint.y_coord);
        assert_eq!(fetched_keypoint.size, *insert_keypoint.size);
        assert_eq!(fetched_keypoint.angle, *insert_keypoint.angle);
        assert_eq!(fetched_keypoint.response, *insert_keypoint.response);
        assert_eq!(fetched_keypoint.octave, *insert_keypoint.octave);
        assert_eq!(fetched_keypoint.class_id, *insert_keypoint.class_id);
        assert_eq!(
            fetched_keypoint.descriptor[0],
            insert_keypoint.clone().descriptor[0]
        );
        assert_eq!(fetched_keypoint.image_id, *insert_keypoint.image_id);
    }

    #[test]
    fn keypoint_fetching_id() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        generate_images_in_database(connection, 1);

        let insert_keypoint = models::InsertKeypoint {
            x_coord: &1.0,
            y_coord: &1.5,
            size: &2.0,
            angle: &2.5,
            response: &3.0,
            octave: &4,
            class_id: &5,
            descriptor: &[6_u8],
            image_id: &1,
        };

        diesel::insert_into(crate::schema::keypoint::table)
            .values(&insert_keypoint)
            .returning(models::Keypoint::as_returning())
            .get_result(connection)
            .expect("Erorr saving new keypoint");

        let fetched_keypoint = Keypoint::read_keypoint_from_id(connection, 1)
            .expect("Could not read keypoint from database");

        assert_eq!(fetched_keypoint.x_coord, *insert_keypoint.x_coord);
        assert_eq!(fetched_keypoint.y_coord, *insert_keypoint.y_coord);
        assert_eq!(fetched_keypoint.size, *insert_keypoint.size);
        assert_eq!(fetched_keypoint.angle, *insert_keypoint.angle);
        assert_eq!(fetched_keypoint.response, *insert_keypoint.response);
        assert_eq!(fetched_keypoint.octave, *insert_keypoint.octave);
        assert_eq!(fetched_keypoint.class_id, *insert_keypoint.class_id);
        assert_eq!(
            fetched_keypoint.descriptor[0],
            insert_keypoint.clone().descriptor[0]
        );
        assert_eq!(fetched_keypoint.image_id, *insert_keypoint.image_id);
    }

    #[test]
    fn keypoint_fetching_id_not_available() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        let fetched_keypoint = Keypoint::read_keypoint_from_id(connection, 1);

        assert!(fetched_keypoint.is_err_and(|e| e.eq(&DieselError::NotFound)));
    }

    #[test]
    fn keypoint_fetching_image_id() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        generate_images_in_database(connection, 3);

        let keypoint_vec = vec![
            models::InsertKeypoint {
                x_coord: &1.0,
                y_coord: &1.5,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &1,
            },
            models::InsertKeypoint {
                x_coord: &2.0,
                y_coord: &3.5,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &2,
            },
            models::InsertKeypoint {
                x_coord: &5.3,
                y_coord: &3.5,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &3,
            },
            models::InsertKeypoint {
                x_coord: &2.0,
                y_coord: &9.0,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &3,
            },
        ];

        keypoint_vec
            .clone()
            .into_iter()
            .for_each(|single_keypoint| {
                diesel::insert_into(crate::schema::keypoint::table)
                    .values(&single_keypoint)
                    .returning(models::Keypoint::as_returning())
                    .get_result(connection)
                    .expect("Error saving new keypoint");
            });

        let fetched_keypoints = Keypoint::read_keypoints_from_image_id(connection, 3)
            .expect("Could not find keypoints");

        assert_eq!(
            fetched_keypoints[0].x_coord,
            *keypoint_vec.clone()[2].x_coord
        );
        assert_eq!(
            fetched_keypoints[0].y_coord,
            *keypoint_vec.clone()[2].y_coord
        );
        assert_eq!(
            fetched_keypoints[1].x_coord,
            *keypoint_vec.clone()[3].x_coord
        );
        assert_eq!(
            fetched_keypoints[1].y_coord,
            *keypoint_vec.clone()[3].y_coord
        );
    }

    #[test]
    fn keypoints_fetched_from_lod() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        let insert_image = InsertImage {
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
            .expect("Could not insert image into database");

        let keypoint_vec = vec![
            models::InsertKeypoint {
                x_coord: &1.0,
                y_coord: &1.5,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &1,
            },
            models::InsertKeypoint {
                x_coord: &2.0,
                y_coord: &3.5,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &1,
            },
            models::InsertKeypoint {
                x_coord: &5.3,
                y_coord: &3.5,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &1,
            },
            models::InsertKeypoint {
                x_coord: &2.0,
                y_coord: &9.0,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &1,
            },
        ];

        keypoint_vec
            .clone()
            .into_iter()
            .for_each(|single_keypoint| {
                diesel::insert_into(crate::schema::keypoint::table)
                    .values(&single_keypoint)
                    .returning(models::Keypoint::as_returning())
                    .get_result(connection)
                    .expect("Error saving new keypoint");
            });

        let fetched_keypoints = Keypoint::read_keypoints_from_lod(connection, 1);

        assert!(fetched_keypoints.is_ok_and(|keypoint_result| keypoint_result.len() == 4));
    }

    #[test]
    fn keypoint_fetching_coordinates() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        let image_vec = vec![
            InsertImage {
                x_start: &1,
                y_start: &1,
                x_end: &3,
                y_end: &4,
                level_of_detail: &1,
            },
            InsertImage {
                x_start: &1,
                y_start: &2,
                x_end: &3,
                y_end: &4,
                level_of_detail: &2,
            },
        ];

        image_vec.into_iter().for_each(|single_image| {
            diesel::insert_into(crate::schema::ref_image::table)
                .values(&single_image)
                .returning(models::Image::as_returning())
                .get_result(connection)
                .expect("Error saving new image.");
        });

        let keypoint_vec = vec![
            models::InsertKeypoint {
                x_coord: &1.0,
                y_coord: &1.5,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &1,
            },
            models::InsertKeypoint {
                x_coord: &2.0,
                y_coord: &3.5,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &1,
            },
            models::InsertKeypoint {
                x_coord: &5.3,
                y_coord: &3.5,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &1,
            },
            models::InsertKeypoint {
                x_coord: &2.0,
                y_coord: &9.0,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &2,
            },
        ];

        keypoint_vec.into_iter().for_each(|single_keypoint| {
            diesel::insert_into(crate::schema::keypoint::table)
                .values(&single_keypoint)
                .returning(models::Keypoint::as_returning())
                .get_result(connection)
                .expect("Error saving new keypoint");
        });

        let fetched_keypoints =
            Keypoint::read_keypoints_from_coordinates(connection, 1.0, 1.0, 2.5, 3.5, 1)
                .expect("Could not fetch keypoints");

        assert_eq!(fetched_keypoints[0].id, 1);
        assert_eq!(fetched_keypoints[1].id, 2);
        assert_eq!(fetched_keypoints.len(), 2);
    }

    #[test]
    fn deleting_keypoint() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        generate_images_in_database(connection, 1);

        let keypoint_vec = vec![
            models::InsertKeypoint {
                x_coord: &1.0,
                y_coord: &1.5,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &1,
            },
            models::InsertKeypoint {
                x_coord: &2.0,
                y_coord: &3.5,
                size: &2.0,
                angle: &2.5,
                response: &3.0,
                octave: &4,
                class_id: &5,
                descriptor: &[6_u8],
                image_id: &1,
            },
        ];

        keypoint_vec.into_iter().for_each(|single_keypoint| {
            diesel::insert_into(crate::schema::keypoint::table)
                .values(&single_keypoint)
                .returning(models::Keypoint::as_returning())
                .get_result(connection)
                .expect("Error saving new keypoint");
        });

        let func_result = Keypoint::delete_keypoint(connection, 1);

        let db_result = keypoint
            .select(models::Keypoint::as_select())
            .load(connection)
            .expect("Error loading images");

        assert_eq!(db_result[0].id, 2);
        assert!(func_result.is_ok());
    }
}
