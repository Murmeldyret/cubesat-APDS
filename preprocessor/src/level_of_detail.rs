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
