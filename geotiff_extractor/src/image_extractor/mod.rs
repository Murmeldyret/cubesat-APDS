use gdal::errors;
use gdal::Dataset;
fn import_datasets(path: &[&str]) -> Result<Vec<Dataset>, errors::GdalError> {
    Ok(path.into_iter().map(|p| Dataset::open(p)?).collect())
}

fn mosaic_datasets(_datasets: &[Dataset]) {
    todo!()
}

