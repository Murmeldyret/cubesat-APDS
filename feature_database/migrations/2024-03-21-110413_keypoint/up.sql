-- Your SQL goes here
CREATE TABLE "keypoint" (
  "id" integer PRIMARY KEY,
  "x_coord" float,
  "y_coord" float,
  "size" float,
  "angle" float,
  "response" float,
  "octave" integer,
  "class_id" integer,
  "image_id" integer
);

ALTER TABLE "keypoints" ADD FOREIGN KEY ("image_id") REFERENCES "image" ("id");