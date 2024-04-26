use crate::models;
use crate::schema::keypoint::dsl;
use diesel::pg::PgConnection;
use diesel::prelude::*;
use diesel::result::Error as DieselError;

pub enum Keypoint<'a> {
    One(models::InsertKeypoint<'a>),
    Multiple(Vec<models::InsertKeypoint<'a>>),
}

const OPENCV_KEYPOINT_LIMIT: i64 = 2_i64.pow(18) - 1;

impl<'a> KeypointDatabase for Keypoint<'a> {
    fn create_keypoint(
        conn: &mut PgConnection,
        input_keypoint: Keypoint,
    ) -> Result<(), DieselError> {
        match input_keypoint {
            Keypoint::One(single_image) => create_keypoint_in_database(conn, &[single_image])?,
            Keypoint::Multiple(multiple_images) => {
                create_keypoint_in_database(conn, &multiple_images)?
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
            .order(dsl::response.desc())
            .limit(OPENCV_KEYPOINT_LIMIT)
            .select(models::Keypoint::as_select())
            .load(conn)
    }

    fn read_keypoints_from_lod(
        conn: &mut PgConnection,
        level_of_detail: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError> {
        use crate::schema::ref_image;

        let keypoints_vec: Vec<models::Keypoint> = dsl::keypoint
            .inner_join(ref_image::dsl::ref_image)
            .filter(ref_image::dsl::level_of_detail.eq(level_of_detail))
            .order(dsl::response.desc())
            .limit(OPENCV_KEYPOINT_LIMIT)
            .select(models::Keypoint::as_select())
            .load(conn)?;

        Ok(keypoints_vec)
    }

    fn read_keypoints_from_coordinates(
        conn: &mut PgConnection,
        x_start: f32,
        y_start: f32,
        x_end: f32,
        y_end: f32,
        level_of_detail: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError> {
        use crate::schema::ref_image;

        let keypoints_vec: Vec<models::Keypoint> = dsl::keypoint
            .inner_join(ref_image::dsl::ref_image)
            .filter(ref_image::dsl::level_of_detail.eq(level_of_detail))
            .filter(dsl::x_coord.ge(x_start.floor()))
            .filter(dsl::x_coord.le(x_end.ceil()))
            .filter(dsl::y_coord.ge(y_start.floor()))
            .filter(dsl::y_coord.le(y_end.ceil()))
            .order(dsl::response.desc())
            .limit(OPENCV_KEYPOINT_LIMIT)
            .select(models::Keypoint::as_select())
            .load(conn)?;

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
    input_keypoint: &[models::InsertKeypoint],
) -> Result<(), DieselError> {
    diesel::insert_into(crate::schema::keypoint::table)
        .values(input_keypoint)
        .execute(connection)?;

    Ok(())
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
        x_start: f32,
        y_start: f32,
        x_end: f32,
        y_end: f32,
        level_of_detail: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError>;
    fn delete_keypoint(conn: &mut PgConnection, id: i32) -> Result<(), DieselError>;
}

#[cfg(test)]
mod tests {
    use self::models::InsertImage;
    use self::models::InsertKeypoint;

    use super::*;
    use crate::db_helpers::{obtain_lock, setup_database};
    use crate::schema::keypoint::dsl::*;

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

    struct TestKeypoint {
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

    fn generate_keypoints_in_database(connection: &mut PgConnection, amount: i64) {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        let insert_image = InsertImage {
                x_start: &rng.gen(),
                y_start: &rng.gen(),
                x_end: &rng.gen(),
                y_end: &rng.gen(),
                level_of_detail: &1,
            };

            diesel::insert_into(crate::schema::ref_image::table)
                .values(insert_image)
                .returning(models::Image::as_returning())
                .get_result(connection)
                .expect("Could not generate images to the database");

                let mut keypoints: Vec<TestKeypoint> = Vec::with_capacity(amount.try_into().unwrap());

        for _i in 0..amount {
                keypoints.push(TestKeypoint {
                x_coord: 1.0,
                y_coord: 1.0,
                size: 1.0,
                angle: 1.0,
                response: 1.0,
                octave: 1,
                class_id: 1,
                descriptor: [6_u8].to_vec(),
                image_id: 1,
            });}

        let mut insert_keypoints: Vec<InsertKeypoint> = Vec::with_capacity(amount.try_into().unwrap());

        for int_keypoint in &keypoints {
        insert_keypoints.push(models::InsertKeypoint {
            x_coord: &int_keypoint.x_coord,
            y_coord: &int_keypoint.y_coord,
            size: &int_keypoint.size,
            angle: &int_keypoint.angle,
            response: &int_keypoint.response,
            octave: &int_keypoint.octave,
            class_id: &int_keypoint.class_id,
            descriptor: &int_keypoint.descriptor,
            image_id: &int_keypoint.image_id,
        });
        if insert_keypoints.len() == 1024 {
            let db_insert_keypoints = Keypoint::Multiple(insert_keypoints.clone());

            Keypoint::create_keypoint(connection, db_insert_keypoints).unwrap();

            insert_keypoints.clear();
        }

        let db_insert_keypoints = Keypoint::Multiple(insert_keypoints.clone());

            Keypoint::create_keypoint(connection, db_insert_keypoints).unwrap();

            insert_keypoints.clear();

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

    // #[ignore = "Very slow"]
    #[test]
    fn opencv_limit_enforcement() {
        let _lock = obtain_lock();
        let connection = &mut setup_database();

        generate_keypoints_in_database(connection, OPENCV_KEYPOINT_LIMIT + 1000);

        let keypoints = Keypoint::read_keypoints_from_lod(connection, 1).unwrap();

        assert_eq!(keypoints.len(), OPENCV_KEYPOINT_LIMIT.try_into().unwrap());
    }
}
