-- Your SQL goes here
CREATE TABLE "descriptor" (
  "id" SERIAL PRIMARY KEY,
  "value" bytea NOT NULL
);

ALTER TABLE "descriptor" ADD FOREIGN KEY ("id") REFERENCES "keypoint" ("id");