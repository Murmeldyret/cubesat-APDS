use gdal::errors;
use gdal::raster::{RasterCreationOption, StatisticsMinMax};
use gdal::Dataset;

use std::path::Path;

use gdal::programs::raster::build_vrt;

pub struct RawDataset {
    pub datasets: Vec<Dataset>,
}

pub struct MosaicedDataset {
    pub dataset: Dataset,
}

pub trait Datasets {
    fn import_datasets(paths: &[&str]) -> Result<RawDataset, errors::GdalError>;
    fn mosaic_datasets(&self) -> Result<MosaicedDataset, errors::GdalError>;
    fn datasets_min_max(&self) -> BandsMinMax;
}

#[derive(Debug)]
pub struct BandsMinMax {
    pub red_min: f64,
    pub red_max: f64,
    pub green_min: f64,
    pub green_max: f64,
    pub blue_min: f64,
    pub blue_max: f64,
}

impl Datasets for RawDataset {
    fn import_datasets(paths: &[&str]) -> Result<RawDataset, errors::GdalError> {
        let ds = paths.into_iter().map(|p| Dataset::open(p)).collect();
        let unwrapped_data = match ds {
            Ok(data) => data,
            Err(e) => return Err(e),
        };
        Ok(RawDataset {
            datasets: unwrapped_data,
        })
    }

    fn mosaic_datasets(&self) -> Result<MosaicedDataset, errors::GdalError> {
        let result_vrt = build_vrt(Some(Path::new("ressources/dataset")), &self.datasets, None)?;

        let create_options = vec![
            RasterCreationOption {
                key: "COMPRESS",
                value: "LZW",
            },
            RasterCreationOption {
                key: "PREDICTOR",
                value: "YES",
            },
            RasterCreationOption {
                key: "BIGTIFF",
                value: "YES",
            },
        ];

        result_vrt.create_copy(
            &gdal::DriverManager::get_driver_by_name("COG")?,
            Path::new("ressources/dataset/"),
            &create_options,
        )?;

        Ok(MosaicedDataset {
            dataset: result_vrt,
        })
    }

    fn datasets_min_max(&self) -> BandsMinMax {
        let datasets = &self.datasets;

        let amount_images = datasets.len() as f64;

        let sum_min_max: Vec<StatisticsMinMax> = (1..4)
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

        let avg_min_max: Vec<StatisticsMinMax> = sum_min_max
            .into_iter()
            .map(|min_max| StatisticsMinMax {
                min: min_max.min / amount_images,
                max: min_max.max / amount_images,
            })
            .collect();

        BandsMinMax {
            red_min: avg_min_max[0].min,
            red_max: avg_min_max[0].max,
            green_min: avg_min_max[1].min,
            green_max: avg_min_max[1].max,
            blue_min: avg_min_max[2].min,
            blue_max: avg_min_max[2].max,
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

        let result = RawDataset::import_datasets(&wrong_paths);

        assert!(result.is_err());
    }

    #[test]
    fn import_dataset_exists() {
        let manifest =
            env::var("CARGO_MANIFEST_DIR").expect("Expected CARGO_MANIFEST_DIR to be set");

        let path = format!(
            "{}/../ressources/test/Geotiff/MOSAIC-0000018944-0000037888.tif",
            manifest
        ); //TODO: Fix path

        let path_vec = vec![path.as_str()];

        let result = RawDataset::import_datasets(&path_vec);

        assert!(result.is_ok_and(|d| d.datasets.capacity() == 1))
    }

    #[test]
    fn combining_dataset() {
        let manifest =
            env::var("CARGO_MANIFEST_DIR").expect("Expected CARGO_MANIFEST_DIR to be set");

        let path1 = format!(
            "{}/../ressources/test/Geotiff/MOSAIC-0000018944-0000037888.tif",
            manifest
        );
        let path2 = format!(
            "{}/../ressources/test/Geotiff/MOSAIC-0000018944-0000018944.tif",
            manifest
        );

        let paths = vec![path1, path2];

        let datasets: Vec<Dataset> = paths
            .into_iter()
            .map(|p| Dataset::open(p))
            .collect::<Result<Vec<Dataset>, errors::GdalError>>()
            .expect("Could not open test files.");

        let datasets = RawDataset { datasets };

        let result = RawDataset::mosaic_datasets(&datasets);

        assert!(result.is_ok())
    }

    #[test]
    fn find_min_max_dataset() {
        let mut current_dir = env::current_dir().expect("Current directory not set.");

        current_dir.pop();

        current_dir.push("ressources/test/Geotiff/MOSAIC-0000018944-0000037888.tif");

        dbg!(&current_dir);

        let ds = Dataset::open(current_dir.as_path()).expect("Could not open dataset");

        let dataset: Vec<Dataset> = vec![ds];

        let datasets = RawDataset { datasets: dataset };

        let result = RawDataset::datasets_min_max(&datasets);

        assert_eq!(0.0017, (result.red_min * 10000.0).round() / 10000.0);
    }

    #[test]
    fn find_min_max_multiple_dataset() {
        let mut current_dir = env::current_dir().expect("Current directory not set.");

        current_dir.pop();

        let mut path1 = current_dir.clone();
        path1.push("ressources/test/Geotiff/MOSAIC-0000018944-0000037888.tif");

        let mut path2 = current_dir.clone();
        path2.push("ressources/test/Geotiff/MOSAIC-0000018944-0000018944.tif");

        let dataset1 = Dataset::open(path1.as_path()).expect("Could not open dataset");
        let dataset2 = Dataset::open(path2.as_path()).expect("Could not open dataset");

        let datasets = RawDataset {
            datasets: vec![dataset1, dataset2],
        };

        let result = RawDataset::datasets_min_max(&datasets);

        assert_eq!(0.0018, (result.red_min * 10000.0).round() / 10000.0);
    }
}

