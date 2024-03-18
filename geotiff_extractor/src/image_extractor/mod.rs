use gdal::errors;
use gdal::raster::{RasterCreationOption, StatisticsMinMax};
use gdal::Dataset;

use std::path::Path;

use gdal::raster::ResampleAlg;

use gdal::programs::raster::build_vrt;

#[cfg(test)]
use mockall::{automock, mock, predicate::*};

// A struct for handling raw datasets from disk in Geotiff format
pub struct RawDataset {
    pub datasets: Vec<Dataset>,
}
pub struct DatasetOptions {
    pub scaling: Option<(i64, i64)>,
}

impl Default for DatasetOptions {
    fn default() -> Self {
        Self {
            scaling: Some((1024, 1024)),
        }
    }
}

// The converted mosaic dataset in COG format
pub struct MosaicedDataset {
    pub dataset: Dataset,
    pub options: DatasetOptions,
}

#[cfg_attr(test, automock)]
pub trait Datasets {
    fn import_datasets(paths: &[String]) -> Result<RawDataset, errors::GdalError>;
    fn to_mosaic_dataset(&self, output_path: &Path) -> Result<MosaicedDataset, errors::GdalError>;
}

#[cfg_attr(test, automock)]
pub trait MosaicDataset {
    fn import_mosaic_dataset(path: &str) -> Result<MosaicedDataset, errors::GdalError>;
    fn datasets_min_max(&self) -> Result<BandsMinMax, errors::GdalError>;
    fn get_dimensions(&self) -> Result<(i64, i64), errors::GdalError>;
    fn set_scaling(&self, dimensions: (i64, i64));
    fn to_rgb(
        &self,
        window: (isize, isize),
        window_size: (usize, usize),
        size: (usize, usize),
    ) -> Result<Vec<rgb::RGBA8>, errors::GdalError>;
    fn detect_nodata(&self) -> bool;
    fn fill_nodata(&self);
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

#[derive(Debug, Clone)]
pub enum PixelConversion {
    GammaOutOfRange,
    FloatToIntegerError,
    NotANumber,
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
    fn to_mosaic_dataset(&self, output_path: &Path) -> Result<MosaicedDataset, errors::GdalError> {
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

        Ok(MosaicedDataset {
            dataset: mosaic,
            options: DatasetOptions {
                scaling: Some((1024, 1024)),
            },
        })
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

    fn get_dimensions(&self) -> Result<(i64, i64), errors::GdalError> {
        todo!()
    }

    fn set_scaling(&self, dimensions: (i64, i64)) {
        todo!()
    }

    fn to_rgb(
        &self,
        window: (isize, isize),
        window_size: (usize, usize),
        size: (usize, usize),
    ) -> Result<Vec<rgb::RGBA8>, errors::GdalError> {
        let red_band = self.dataset.rasterband(1)?;
        let green_band = self.dataset.rasterband(2)?;
        let blue_band = self.dataset.rasterband(3)?;

        let min_max = self.datasets_min_max()?;

        let red_vec: Result<Vec<u8>, PixelConversion> = red_band
            .read_as::<f32>(window, window_size, size, Some(ResampleAlg::Lanczos))?
            .data
            .into_iter()
            .map(|p| f32_to_u8(p, min_max.red_min as f32, min_max.red_max as f32))
            .collect();

        let red_vec = match red_vec {
            Ok(vec) => vec,
            Err(_) => return Err(errors::GdalError::BadArgument("".to_string())),
        };

        dbg!(red_vec);

        todo!()
    }

    fn detect_nodata(&self) -> bool {
        todo!()
    }

    fn fill_nodata(&self) {
        todo!()
    }

    fn import_mosaic_dataset(path: &str) -> Result<MosaicedDataset, errors::GdalError> {
        todo!()
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

fn gamma_correction(input_value: f32) -> Result<f32, PixelConversion> {
    if input_value < 0.0 || input_value > 1.0 {
        return Err(PixelConversion::GammaOutOfRange);
    }

    Ok(input_value.powf(2.2))
}

fn f32_to_u8(input_value: f32, min: f32, max: f32) -> Result<u8, PixelConversion> {
    if input_value.is_nan() {
        return Err(PixelConversion::NotANumber);
    }

    let float = (input_value - min) / (max - min);

    let normal_float = gamma_correction(float)?;

    Ok((normal_float * 255.0).round() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn import_dataset_missing() {
        let wrong_paths = vec![String::from("/Nowhere")];

        let result = RawDataset::import_datasets(&wrong_paths);

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

        assert!(result.is_ok_and(|d| d.datasets[0].raster_size() == (7309, 4322)));
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

        let result = datasets.to_mosaic_dataset(output_path.as_path());

        assert!(result.is_ok());
    }

    #[test]
    fn find_min_max_dataset() {
        let mut current_dir = env::current_dir().expect("Current directory not set.");

        current_dir.pop();

        current_dir.push("resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif");

        dbg!(&current_dir);

        let ds = Dataset::open(current_dir.as_path()).expect("Could not open dataset");

        let dataset = MosaicedDataset {
            dataset: ds,
            options: DatasetOptions { scaling: None },
        };

        let result = MosaicDataset::datasets_min_max(&dataset);

        assert_eq!(
            0.0017,
            (result.unwrap().red_min * 10000.0).round() / 10000.0
        );
    }

    #[test]
    fn gamma_correct_input() {
        let input_value = 0.5;

        let output_value = gamma_correction(input_value);

        dbg!(&output_value.clone().unwrap());

        assert!(output_value.is_ok_and(|result| result == 0.21763763));
    }

    #[test]
    fn gamma_value_above_1() {
        let input_value = 1.5;

        let output_value = gamma_correction(input_value);

        assert!(output_value.is_err());
    }

    #[test]
    fn gamma_value_below_0() {
        let input_value = -0.5;

        let output_value = gamma_correction(input_value);

        assert!(output_value.is_err());
    }

    //TODO: Gamma skal mockes, men det gidder jeg ikke lige nu.
    #[test]
    fn convert_f32_to_u8_success() {
        let input_value = 0.2;
        let min = 0.1;
        let max = 0.3;

        let output_value = f32_to_u8(input_value, min, max);

        assert!(output_value.is_ok_and(|result| result == 55));
    }

    #[test]
    fn convert_f32_to_u8_nan() {
        let input_value = f32::NAN;
        let min = 0.1;
        let max = 0.3;

        let output_value = f32_to_u8(input_value, min, max);

        assert!(output_value.is_err_and(|error| match error {
            PixelConversion::NotANumber => true,
            _ => false,
        }));
    }

    #[test]
    fn dataset_to_rgb() {
        let mut current_dir = env::current_dir().expect("Current directory not set.");

        current_dir.pop();

        current_dir.push("resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif");

        dbg!(&current_dir);

        let ds = Dataset::open(current_dir.as_path()).expect("Could not open dataset");

        let dataset = MosaicedDataset {
            dataset: ds,
            options: DatasetOptions { scaling: None },
        };

        let window_size = dataset.dataset.raster_size();

        let image: Result<Vec<rgb::RGBA8>, _> = dataset.to_rgb((0, 0), window_size, (1024, 1024));

        assert!(image.is_ok_and(|image_vec| image_vec.len() == 1024 * 1024));
    }
}

