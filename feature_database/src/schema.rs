// @generated automatically by Diesel CLI.

diesel::table! {
    keypoint (id) {
        id -> Int4,
        x_coord -> Float8,
        y_coord -> Float8,
        size -> Float8,
        angle -> Float8,
        response -> Float8,
        octave -> Int4,
        class_id -> Int4,
        descriptor -> Bytea,
        image_id -> Int4,
    }
}

diesel::table! {
    ref_image (id) {
        id -> Int4,
        x_start -> Int4,
        y_start -> Int4,
        x_end -> Int4,
        y_end -> Int4,
        level_of_detail -> Int4,
    }
}

diesel::joinable!(keypoint -> ref_image (image_id));

diesel::allow_tables_to_appear_in_same_query!(keypoint, ref_image,);
