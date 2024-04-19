// @generated automatically by Diesel CLI.

diesel::table! {
    keypoint (id) {
        id -> Int4,
        x_coord -> Float4,
        y_coord -> Float4,
        size -> Float4,
        angle -> Float4,
        response -> Float4,
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
