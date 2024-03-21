// @generated automatically by Diesel CLI.

diesel::table! {
    descriptors (id) {
        id -> Int4,
        value -> Nullable<Bytea>,
    }
}

diesel::table! {
    image (id) {
        id -> Int4,
        x_start -> Nullable<Int4>,
        y_start -> Nullable<Int4>,
        x_end -> Nullable<Int4>,
        y_end -> Nullable<Int4>,
        level_of_detail -> Nullable<Int4>,
    }
}

diesel::table! {
    keypoints (id) {
        id -> Int4,
        x_coord -> Nullable<Float8>,
        y_coord -> Nullable<Float8>,
        size -> Nullable<Float8>,
        angle -> Nullable<Float8>,
        response -> Nullable<Float8>,
        octave -> Nullable<Int4>,
        class_id -> Nullable<Int4>,
        image_id -> Nullable<Int4>,
    }
}

diesel::table! {
    level_of_detail (id) {
        id -> Int4,
    }
}

diesel::joinable!(descriptors -> keypoints (id));
diesel::joinable!(keypoints -> image (image_id));

diesel::allow_tables_to_appear_in_same_query!(
    descriptors,
    image,
    keypoints,
    level_of_detail,
);
