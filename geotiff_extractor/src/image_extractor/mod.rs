use gdal::errors;
use gdal::Dataset;


struct GDALDataset {
    pub dataset: Vec<Dataset>,
}

pub trait Datasets {
    fn import_datasets(paths: &[&str]) -> Result<GDALDataset, errors::GdalError>;
    fn mosaic_datasets(_datasets: &[GDALDataset]) -> Result<GDALDataset, errors::GdalError>;
    fn datasets_min_max(&self) -> BandsMinMax;

}


struct BandsMinMax {
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
            Err(e) => return Err(e)
        };
        Ok(GDALDataset { dataset: unwrapped_data })
    }

    fn mosaic_datasets(_datasets: &[GDALDataset]) -> Result<GDALDataset, errors::GdalError> {
        todo!()
    }

    fn datasets_min_max(&self) -> BandsMinMax {
        todo!()
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
        let manifest = env::var("CARGO_MANIFEST_DIR").expect("Expected CARGO_MANIFEST_DIR to be set");

        let path = format!("{}/../resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif", manifest); //TODO: Fix path

        let path_vec = vec![path.as_str()];

        let result = GDALDataset::import_datasets(&path_vec);

        assert!(result.is_ok_and(|d| d.dataset.capacity() == 1))
    }

    #[test]
    fn combining_dataset() {

        let manifest = env::var("").expect("Expected CARGO_MANIFEST_DIR to be set");

        let path1 = format!("{}/../resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif", manifest);
        let path2 = format!("{}/../resources/test/Geotiff/MOSAIC-0000018944-0000018944.tif", manifest);

        let paths = vec![path1, path2];

        let datasets: Vec<Dataset> = paths.into_iter().map(|p| Dataset::open(p)).collect::<Result<Vec<Dataset>, errors::GdalError>>().expect("Could not open test files.");

        let result = mosaic_datasets(&datasets);

        assert!(result.is_ok())
    }
}
