-- Your SQL goes here
CREATE TABLE "image" (
  "id" integer PRIMARY KEY,
  "x_start" integer,
  "y_start" integer,
  "x_end" integer,
  "y_end" integer,
  "level_of_detail" integer
);

ALTER TABLE "image" ADD FOREIGN KEY ("level_of_detail") REFERENCES "level_of_detail" ("id");