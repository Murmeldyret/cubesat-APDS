use gdal::errors;
use gdal::raster::StatisticsMinMax;
use gdal::Dataset;

use rayon::prelude::*;

struct GDALDataset {
    pub dataset: Vec<Dataset>,
}

pub trait Datasets {
    fn import_datasets(paths: &[&str]) -> Result<GDALDataset, errors::GdalError>;
    fn mosaic_datasets(_datasets: &GDALDataset) -> Result<GDALDataset, errors::GdalError>;
    fn datasets_min_max(&self) -> BandsMinMax;
}

pub struct BandsMinMax {
    red_min: f64,
    red_max: f64,
    green_min: f64,
    green_max: f64,
    blue_min: f64,
    blue_max: f64,
}

impl Datasets for GDALDataset {
    fn import_datasets(paths: &[&str]) -> Result<GDALDataset, errors::GdalError> {
        let ds = paths.into_iter().map(|p| Dataset::open(p)).collect();
        let unwrapped_data = match ds {
            Ok(data) => data,
            Err(e) => return Err(e),
        };
        Ok(GDALDataset {
            dataset: unwrapped_data,
        })
    }

    fn mosaic_datasets(_datasets: &GDALDataset) -> Result<GDALDataset, errors::GdalError> {
        todo!()
    }

    fn datasets_min_max(&self) -> BandsMinMax {
        let datasets = &self.dataset;

        let amount_images = datasets.len();

        let something: Vec<StatisticsMinMax> = (1..4)
            .into_iter()
            .map(|i| {
                datasets
                    .into_iter()
                    .fold(StatisticsMinMax { min: 0.0, max: 0.0 }, |min_max, ds| {
                        let ds_min_max = ds
                            .rasterband(i)
                            .unwrap()
                            .compute_raster_min_max(true)
                            .unwrap();
                        StatisticsMinMax {
                            min: ds_min_max.min + min_max.min,
                            max: ds_min_max.max + min_max.max,
                        }
                    })
            })
            .collect();

        dbg!(&something);

        BandsMinMax {
            red_min: something[0].min,
            red_max: something[0].max,
            green_min: something[1].min,
            green_max: something[1].max,
            blue_min: something[2].min,
            blue_max: something[2].max,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn import_dataset_missing() {
        let wrong_paths = vec!["/somewhere/where/nothing/exists"];

        let result = GDALDataset::import_datasets(&wrong_paths);

        assert!(result.is_err());
    }

    #[test]
    fn import_dataset_exists() {
        let manifest =
            env::var("CARGO_MANIFEST_DIR").expect("Expected CARGO_MANIFEST_DIR to be set");

        let path = format!(
            "{}/../resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif",
            manifest
        ); //TODO: Fix path

        let path_vec = vec![path.as_str()];

        let result = GDALDataset::import_datasets(&path_vec);

        assert!(result.is_ok_and(|d| d.dataset.capacity() == 1))
    }

    #[test]
    fn combining_dataset() {
        let manifest =
            env::var("CARGO_MANIFEST_DIR").expect("Expected CARGO_MANIFEST_DIR to be set");

        let path1 = format!(
            "{}/../resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif",
            manifest
        );
        let path2 = format!(
            "{}/../resources/test/Geotiff/MOSAIC-0000018944-0000018944.tif",
            manifest
        );

        let paths = vec![path1, path2];

        let datasets: Vec<Dataset> = paths
            .into_iter()
            .map(|p| Dataset::open(p))
            .collect::<Result<Vec<Dataset>, errors::GdalError>>()
            .expect("Could not open test files.");

        let datasets = GDALDataset { dataset: datasets };

        let result = GDALDataset::mosaic_datasets(&datasets);

        assert!(result.is_ok())
    }

    #[test]
    fn find_min_max_dataset() {
        let mut current_dir = env::current_dir().expect("Current directory not set.");

        current_dir.pop();

        current_dir.push("resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif");

        dbg!(&current_dir);

        let ds = Dataset::open(current_dir.as_path()).expect("Could not open dataset");

        let dataset: Vec<Dataset> = vec![ds];

        let datasets = GDALDataset { dataset: dataset };

        let result = GDALDataset::datasets_min_max(&datasets);

        assert_eq!(0.0017, (result.red_min * 10000.0).round() / 10000.0);
    }

    #[test]
    fn find_min_max_multiple_dataset() {
        let mut current_dir = env::current_dir().expect("Current directory not set.");

        current_dir.pop();

        let mut path1 = current_dir.clone();
        path1.push("resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif");

        let mut path2 = current_dir.clone();
        path2.push("resources/test/Geotiff/MOSAIC-0000018944-0000018944.tif");

        let dataset1 = Dataset::open(path1.as_path()).expect("Could not open dataset");
        let dataset2 = Dataset::open(path2.as_path()).expect("Could not open dataset");

        let datasets = GDALDataset {
            dataset: vec![dataset1, dataset2],
        };

        let result = GDALDataset::datasets_min_max(&datasets);

        assert_eq!(0.0036, (result.red_min * 10000.0).round() / 10000.0);
    }
}

