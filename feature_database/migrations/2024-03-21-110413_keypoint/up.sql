-- Your SQL goes here

CREATE TABLE "keypoint" (
  "id" SERIAL PRIMARY KEY,
  "x_coord" float NOT NULL,
  "y_coord" float NOT NULL,
  "size" float NOT NULL,
  "angle" float NOT NULL,
  "response" float NOT NULL,
  "octave" integer NOT NULL,
  "class_id" integer NOT NULL,
  "descriptor" bytea NOT NULL,
  "image_id" integer NOT NULL
);

ALTER TABLE "keypoint" ADD FOREIGN KEY ("image_id") REFERENCES "ref_image" ("id");
