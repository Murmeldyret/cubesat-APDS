use gdal::errors;
use gdal::Dataset;
fn import_datasets(paths: &[&str]) -> Result<Vec<Dataset>, errors::GdalError> {
    paths.into_iter().map(|p| Dataset::open(p)).collect()
}

fn mosaic_datasets(_datasets: &[Dataset]) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn import_dataset_missing() {
        let wrong_paths = vec!["/somewhere/where/nothing/exists"];

        let result = import_datasets(&wrong_paths);

        assert!(result.is_err());
    }

    #[test]
    fn import_dataset_exists() {
        let manifest = env::var("CARGO_MANIFEST_DIR").expect("Expected CARGO_MANIFEST_DIR to be set");

        let path = format!("{}/../resources/test/Geotiff/MOSAIC-0000018944-0000037888.tif", manifest);

        let path_vec = vec![path.as_str()];

        let result = import_datasets(&path_vec);

        assert!(result.is_ok_and(|d| d.capacity() == 1))
    }
}
