-- Your SQL goes here

CREATE TABLE "elevation" (
  "id" SERIAL PRIMARY KEY,
  "height" float NOT NULL
);

CREATE TABLE "elevation_properties" (
  "id" SERIAL PRIMARY KEY,
  "x_size" integer NOT NULL,
  "y_size" integer NOT NULL
);