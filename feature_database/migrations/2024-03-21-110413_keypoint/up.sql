-- Your SQL goes here

CREATE TABLE "keypoint" (
  "id" SERIAL PRIMARY KEY,
  "x_coord" real NOT NULL,
  "y_coord" real NOT NULL,
  "size" real NOT NULL,
  "angle" real NOT NULL,
  "response" real NOT NULL,
  "octave" integer NOT NULL,
  "class_id" integer NOT NULL,
  "descriptor" bytea NOT NULL,
  "image_id" integer NOT NULL
);

ALTER TABLE "keypoint" ADD FOREIGN KEY ("image_id") REFERENCES "ref_image" ("id");
