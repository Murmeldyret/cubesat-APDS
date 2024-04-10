pub fn calculate_amount_of_levels(resolution: u64, tile_size: u64) -> u64 {
    walk_lod(tile_size, resolution) + 1
}

pub fn walk_lod(coverage: u64, tile_size: u64) -> u64 {
    (((tile_size as f64).sqrt()/(coverage as f64).sqrt()).log2()).ceil() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod_total_calc() {
        assert_eq!(calculate_amount_of_levels(1000*1000, 250*250), 3);
    }

    #[test]
    fn walk_layer_to_match_resolution() {
        let scale_factor = walk_lod(250*250, 2000*2000);
        assert_eq!(scale_factor, 3);

        let new_scale_factor = walk_lod(((250*2_i32.pow(scale_factor.try_into().unwrap()))*(250*2_i32.pow(scale_factor.try_into().unwrap()))) as u64, 2000*2000);

        assert_eq!(new_scale_factor, 0);
    }

    #[test]
    fn no_negative_walking() {
        assert_eq!(walk_lod(4000*4000, 1000*1000), 0);
    }
}
