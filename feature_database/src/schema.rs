// @generated automatically by Diesel CLI.

diesel::table! {
    descriptor (id) {
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
    keypoint (id) {
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

diesel::joinable!(descriptor -> keypoint (id));
diesel::joinable!(keypoint -> image (image_id));

diesel::allow_tables_to_appear_in_same_query!(
    descriptor,
    image,
    keypoint,
);
