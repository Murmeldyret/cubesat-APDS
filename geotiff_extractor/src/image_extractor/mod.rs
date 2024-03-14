use gdal::errors;
use gdal::raster::{RasterCreationOption, StatisticsMinMax};
use gdal::Dataset;

use std::path::Path;

use gdal::programs::raster::build_vrt;

// A struct for handling raw datasets from disk in Geotiff format
pub struct RawDataset {
    pub datasets: Vec<Dataset>,
}

// The converted mosaic dataset in COG format
pub struct MosaicedDataset {
    pub dataset: Dataset,
}

pub struct RGBDataset {
    pub dataset: Vec<f32>,
    pub width: i32,
    pub height: i32,
}

pub trait Datasets {
    fn import_datasets(paths: &[String]) -> Result<RawDataset, errors::GdalError>;
    fn mosaic_datasets(&self, output_path: &Path) -> Result<MosaicedDataset, errors::GdalError>;
}

pub trait MosaicDataset {
    fn datasets_min_max(&self) -> Result<BandsMinMax, errors::GdalError>;
    // fn to_rgb(&self)
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
    /// The function will import multiple datasets from a vector of paths.
    /// Providing the function of a slice of [Path]s then it will return a [Result<RawDataset>]
    fn import_datasets(paths: &[String]) -> Result<RawDataset, errors::GdalError> {
        let ds = paths.into_iter().map(|p| Dataset::open(p)).collect(); // Opens every dataset that a path points to.
        let unwrapped_data = match ds {
            Ok(data) => data,
            Err(e) => return Err(e),
        };
        Ok(RawDataset {
            datasets: unwrapped_data,
        })
    }

    /// Returns a mosaic dataset that is a combined version of the [RawDataset] dataset provided.
    fn mosaic_datasets(&self, output_path: &Path) -> Result<MosaicedDataset, errors::GdalError> {
        let mut vrt_path = output_path.to_path_buf();
        let mut cog_path = output_path.to_path_buf();

        vrt_path.push("dataset.vrt");
        cog_path.push("dataset.tif");

        let result_vrt = build_vrt(Some(vrt_path.as_path()), &self.datasets, None)?;

        let create_options = creation_options();

        let mosaic = result_vrt.create_copy(
            &gdal::DriverManager::get_driver_by_name("COG")?,
            cog_path,
            &create_options,
        )?;

        Ok(MosaicedDataset { dataset: mosaic })
    }

    // TODO: Gdal finds the collected min and max of datasets when they are turned into a virtual raster. Therefore this should just lookup the min and max of this raster instead of finding the average.
}

impl MosaicDataset for MosaicedDataset {
    fn datasets_min_max(&self) -> Result<BandsMinMax, errors::GdalError> {
        let dataset = &self.dataset;

        let min_max: Vec<StatisticsMinMax> = (1..4)
            .into_iter()
            .map(|i| {
                let ds_min_max = dataset.rasterband(i)?.compute_raster_min_max(true)?;
                Ok::<StatisticsMinMax, errors::GdalError>(StatisticsMinMax {
                    min: ds_min_max.min,
                    max: ds_min_max.max,
                })
            })
            .collect::<Result<Vec<StatisticsMinMax>, errors::GdalError>>()?;

        Ok(BandsMinMax {
            red_min: min_max[0].min,
            red_max: min_max[0].max,
            green_min: min_max[1].min,
            green_max: min_max[1].max,
            blue_min: min_max[2].min,
            blue_max: min_max[2].max,
        })
    }
}

fn creation_options() -> Vec<RasterCreationOption<'static>> {
    let create_options = vec![
        RasterCreationOption {
            key: "COMPRESS",
            value: "LZW", // Should be changed to ZSTD when it is time to use the system.
        },
        RasterCreationOption {
            key: "PREDICTOR",
            value: "YES",
        },
        RasterCreationOption {
            key: "BIGTIFF",
            value: "YES",
        },
        RasterCreationOption {
            key: "NUM_THREADS",
            value: "ALL_CPUS",
        },
    ];
    create_options
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn import_dataset_missing() {
        let wrong_paths = vec![Path::new("/Nowhere")];

        let string_vec: Vec<String> = wrong_paths
            .into_iter()
            .map(|p| p.to_string_lossy().into())
            .collect();

        let result = RawDataset::import_datasets(&string_vec);

        assert!(result.is_err());
    }

    #[test]
    fn import_dataset_exists() {
        let mut current_dir = env::current_dir().expect("Current directory not set.");

        current_dir.pop();

        dbg!(&current_dir);

        let mut path = current_dir.clone();
        path.push("resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif");

        let path_vec = vec![path.to_string_lossy().into()];

        let result = RawDataset::import_datasets(&path_vec);

        assert!(result.is_ok_and(|d| d.datasets[0].raster_size() == (7309, 4322)))
    }

    #[test]
    fn combining_dataset() {
        //TODO: The test should cleanup after itself. Maybe use tmp files.
        let mut current_dir = env::current_dir().expect("Current directory not set.");

        current_dir.pop();

        let mut path1 = current_dir.clone();
        path1.push("resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif");

        let mut path2 = current_dir.clone();
        path2.push("resources/test/Geotiff/MOSAIC-0000018944-0000018944.tif");

        let mut output_path = current_dir.clone();

        output_path.push("resources/dataset");

        let paths = vec![path1, path2];

        let datasets: Vec<Dataset> = paths
            .into_iter()
            .map(|p| Dataset::open(p.as_path()))
            .collect::<Result<Vec<Dataset>, errors::GdalError>>()
            .expect("Could not open test files.");

        let datasets = RawDataset { datasets };

        let result = datasets.mosaic_datasets(output_path.as_path());

        assert!(result.is_ok())
    }

    #[test]
    fn find_min_max_dataset() {
        let mut current_dir = env::current_dir().expect("Current directory not set.");

        current_dir.pop();

        current_dir.push("resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif");

        dbg!(&current_dir);

        let ds = Dataset::open(current_dir.as_path()).expect("Could not open dataset");

        let dataset = MosaicedDataset { dataset: ds };

        let result = MosaicDataset::datasets_min_max(&dataset);

        assert_eq!(
            0.0017,
            (result.unwrap().red_min * 10000.0).round() / 10000.0
        );
    }
}

