use crate::models;
use diesel::pg::PgConnection;
use diesel::prelude::*;
use diesel::result::Error as DieselError;

pub enum Keypoint<'a> {
    One(models::InsertKeypoint<'a>),
    Multiple(Vec<models::InsertKeypoint<'a>>),
}

impl<'a> KeypointDatabase for Keypoint<'a> {
    fn create_keypoint(conn: &mut PgConnection, keypoint: Keypoint) -> Result<(), DieselError> {
        todo!()
    }

    fn read_keypoint_from_id(
        conn: &mut PgConnection,
        id: i32,
    ) -> Result<models::Keypoint, DieselError> {
        todo!()
    }

    fn read_keypoints_from_image_id(
        conn: &mut PgConnection,
        image_id: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError> {
        todo!()
    }

    fn read_keypoints_from_lod(
        conn: &mut PgConnection,
        level: u32,
    ) -> Result<Vec<models::Keypoint>, DieselError> {
        todo!()
    }

    fn read_keypoints_from_coordinates(
        conn: &mut PgConnection,
        x_start: i32,
        y_start: i32,
        x_end: i32,
        y_end: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError> {
        todo!()
    }

    fn delete_keypoint(conn: &mut PgConnection, id: i32) -> Result<(), DieselError> {
        todo!()
    }
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
    ) -> Result<Vec<models::Keypoint>, DieselError>;
    fn read_keypoints_from_coordinates(
        conn: &mut PgConnection,
        x_start: i32,
        y_start: i32,
        x_end: i32,
        y_end: i32,
    ) -> Result<Vec<models::Keypoint>, DieselError>;
    fn delete_keypoint(conn: &mut PgConnection, id: i32) -> Result<(), DieselError>;
}

#[cfg(test)]
mod tests {
    use self::models::InsertImage;

    use super::*;
    use crate::schema::keypoint::dsl::*;
    use crate::testhelpers::{obtain_lock, setup_test_database};

    fn generate_images_in_database(connection: &mut PgConnection, amount: i32) {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        for i in 0..amount {
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
        let connection = &mut setup_test_database();

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
        let connection = &mut setup_test_database();

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
        let connection = &mut setup_test_database();

        let fetched_keypoint = Keypoint::read_keypoint_from_id(connection, 1);

        assert!(fetched_keypoint.is_err_and(|e| e.eq(&DieselError::NotFound)));
    }

    #[test]
    fn keypoint_fetching_image_id() {
        let _lock = obtain_lock();
        let connection = &mut setup_test_database();

        generate_images_in_database(connection, 3);

        let keypoint_vec = vec![models::InsertKeypoint {
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
            y_coord:&3.5,
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
            y_coord:&3.5,
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
            y_coord:&9.0,
            size: &2.0,
            angle: &2.5,
            response: &3.0,
            octave: &4,
            class_id: &5,
            descriptor: &[6_u8],
            image_id: &3,
        }
        ];

        keypoint_vec.clone().into_iter().for_each(|single_keypoint| {diesel::insert_into(crate::schema::keypoint::table).values(&single_keypoint).returning(models::Keypoint::as_returning()).get_result(connection).expect("Error saving new keypoint");});

        let fetched_keypoints = Keypoint::read_keypoints_from_image_id(connection, 3).expect("Could not find keypoints");

        assert_eq!(fetched_keypoints[0].x_coord, *keypoint_vec.clone()[2].x_coord);
        assert_eq!(fetched_keypoints[0].y_coord, *keypoint_vec.clone()[2].y_coord);
        assert_eq!(fetched_keypoints[1].x_coord, *keypoint_vec.clone()[3].x_coord);
        assert_eq!(fetched_keypoints[1].y_coord, *keypoint_vec.clone()[3].y_coord);
    }

    #[test]
    fn keypoints_fetched_from_lod() {
        let _lock = obtain_lock();
        let connection = &mut setup_test_database();

        let insert_image = InsertImage {
                x_start: &0,
                y_start: &0,
                x_end: &10,
                y_end: &10,
                level_of_detail: &1,
        };

        diesel::insert_into(crate::schema::ref_image::table).values(&insert_image).returning(models::Image::as_returning()).get_result(connection).expect("Could not insert image into database");

        let keypoint_vec = vec![models::InsertKeypoint {
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
            y_coord:&3.5,
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
            y_coord:&3.5,
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
            y_coord:&9.0,
            size: &2.0,
            angle: &2.5,
            response: &3.0,
            octave: &4,
            class_id: &5,
            descriptor: &[6_u8],
            image_id: &1,
        }
        ];

        keypoint_vec.clone().into_iter().for_each(|single_keypoint| {diesel::insert_into(crate::schema::keypoint::table).values(&single_keypoint).returning(models::Keypoint::as_returning()).get_result(connection).expect("Error saving new keypoint");});

        let fetched_keypoints = Keypoint::read_keypoints_from_lod(connection, 1);

        assert!(fetched_keypoints.is_ok_and(|keypoint_result| keypoint_result.len() == 4));
    }
}
