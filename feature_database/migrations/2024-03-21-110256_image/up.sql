-- Your SQL goes here
CREATE TABLE "image" (
  "id" integer PRIMARY KEY,
  "x_start" integer NOT NULL,
  "y_start" integer NOT NULL,
  "x_end" integer NOT NULL,
  "y_end" integer NOT NULL,
  "level_of_detail" integer NOT NULL
);