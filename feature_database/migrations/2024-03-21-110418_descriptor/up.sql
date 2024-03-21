-- Your SQL goes here
CREATE TABLE "descriptor" (
  "id" integer PRIMARY KEY,
  "value" bytea
);

ALTER TABLE "descriptor" ADD FOREIGN KEY ("id") REFERENCES "keypoint" ("id");