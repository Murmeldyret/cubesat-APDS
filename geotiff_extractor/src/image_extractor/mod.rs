use gdal::errors;
use gdal::Dataset;
fn import_datasets(paths: &[&str]) -> Result<Vec<Dataset>, errors::GdalError> {
    paths.into_iter().map(|p| Dataset::open(p)).collect()
}

fn mosaic_datasets(_datasets: &[Dataset]) {
    todo!()
}

