use crate::{Args, DatasetPath};
use geotiff_lib::image_extractor::{Datasets, MosaicDataset, MosaicedDataset, RawDataset};

const MINIMUM_RESOLUTION: i64 = 500;

/// Returns how many layers the lod will consist of.
pub fn calculate_amount_of_levels(reference_image_resolution: u64, tile_resolution: u64) -> u64 {
    walk_lod(tile_resolution, reference_image_resolution) + 1
}

/// Returns an integer that describe how many levels should be traveled to reach the optimal lod.
pub fn walk_lod(pixel_coverage: u64, tile_resolution: u64) -> u64 {
    (((tile_resolution as f64).sqrt() / (pixel_coverage as f64).sqrt()).log2()).ceil() as u64
}

/// Converts lod image coordinates to reference image coordinates.
pub fn calc_offset_from_lod(coordinates: (u64, u64), lod: u64) -> (u64, u64) {
    (
        coordinates.0 * 2_u64.pow(lod.try_into().expect("Casting Error")),
        coordinates.1 * 2_u64.pow(lod.try_into().expect("Casting Error")),
    )
}

pub fn calculate_level_of_detail_resolution(args: &Args) {
    if let DatasetPath::Dataset { path } = &args.dataset_path {
        let dataset = RawDataset::import_datasets(&path).expect("Could not read datasets");

        let resolution = dataset
            .to_vrt_dataset()
            .expect("Could not create vrt")
            .get_dimensions()
            .expect("Could not get resolution");

        print_resolution(resolution.0, resolution.1);
    }

    if let DatasetPath::Mosaic { path } = &args.dataset_path {
        let resolution = MosaicedDataset::import_mosaic_dataset(&path)
            .expect("Could not read dataset")
            .get_dimensions()
            .expect("Could not get resolution");

        print_resolution(resolution.0, resolution.1);
    }
}

pub fn print_resolution(x: i64, y: i64) {
    let mut x = x;
    let mut y = y;
    let mut lod = 0;

    while x >= MINIMUM_RESOLUTION && y >= MINIMUM_RESOLUTION {
        println!("lod: {} | x: {} | y: {}", lod + 1, x, y);

        x = x / 2;
        y = y / 2;
        lod += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod_total_calc() {
        assert_eq!(calculate_amount_of_levels(1000 * 1000, 250 * 250), 3);
    }

    #[test]
    fn walk_layer_to_match_resolution() {
        let scale_factor = walk_lod(250 * 250, 2000 * 2000);
        assert_eq!(scale_factor, 3);

        let new_scale_factor = walk_lod(
            (250 * 2_u64.pow(scale_factor.try_into().unwrap()))
                * (250 * 2_u64.pow(scale_factor.try_into().unwrap())),
            2000 * 2000,
        );

        assert_eq!(new_scale_factor, 0);
    }

    #[test]
    fn no_negative_walking() {
        assert_eq!(walk_lod(4000 * 4000, 1000 * 1000), 0);
    }

    #[test]
    fn offset_calculation_from_lod() {
        assert_eq!(calc_offset_from_lod((1000, 1000), 2), (4000, 4000));
    }

    #[test]
    fn offset_calculation_from_lod_reference() {
        assert_eq!(calc_offset_from_lod((1000, 1000), 0), (1000, 1000));
    }
}
