-- Your SQL goes here
CREATE TABLE "descriptor" (
  "id" integer PRIMARY KEY,
  "value" bytea
);

ALTER TABLE "descriptors" ADD FOREIGN KEY ("id") REFERENCES "keypoints" ("id");