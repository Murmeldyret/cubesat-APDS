-- Your SQL goes here

CREATE TABLE "geotransform" (
  "id" SERIAL PRIMARY KEY,
  "dataset_name" VARCHAR(64) NOT NULL,
  "transform" float[] NOT NULL
);
