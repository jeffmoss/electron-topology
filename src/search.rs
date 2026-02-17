use crate::physics::GeometryCandidate;

pub struct Phase1Params {
    pub rho_min: f64,
    pub rho_max: f64,
    pub num_rho: u32,
    pub max_winding: i32,
}

impl Default for Phase1Params {
    fn default() -> Self {
        Self {
            rho_min: 0.01,
            rho_max: 0.999,
            num_rho: 10_000,
            max_winding: 20,
        }
    }
}

pub struct Phase2Params {
    pub rho_center: f64,
    pub rho_width: f64,
    pub num_rho: u32,
    pub max_winding: i32,
    pub epsilon_range: (f64, f64),
    pub num_epsilon: u32,
}

pub struct Phase3Params {
    pub candidates: Vec<GeometryCandidate>,
    pub quadrature_points: usize,
}

impl Default for Phase3Params {
    fn default() -> Self {
        Self {
            candidates: Vec::new(),
            quadrature_points: 256,
        }
    }
}

/// Generate evenly spaced rho values in [min, max].
pub fn generate_rho_values(min: f64, max: f64, count: u32) -> Vec<f64> {
    if count <= 1 {
        return vec![min];
    }
    let step = (max - min) / (count - 1) as f64;
    (0..count).map(|i| min + i as f64 * step).collect()
}

/// Estimate total number of evaluations for a Phase 1 search.
pub fn estimate_search_size(params: &Phase1Params, num_pq: usize) -> u64 {
    params.num_rho as u64 * num_pq as u64
}

/// Split a total work count into batches of at most `batch_size`, returning
/// (offset, count) pairs.
pub fn generate_batches(total: u64, batch_size: u64) -> Vec<(u64, u64)> {
    let mut batches = Vec::new();
    let mut offset = 0u64;
    while offset < total {
        let count = (total - offset).min(batch_size);
        batches.push((offset, count));
        offset += count;
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_rho_values() {
        let vals = generate_rho_values(0.0, 1.0, 11);
        assert_eq!(vals.len(), 11);
        assert!((vals[0] - 0.0).abs() < 1e-15);
        assert!((vals[10] - 1.0).abs() < 1e-15);
        assert!((vals[5] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_generate_batches() {
        let batches = generate_batches(100, 30);
        assert_eq!(batches, vec![(0, 30), (30, 30), (60, 30), (90, 10)]);
    }

    #[test]
    fn test_generate_batches_exact() {
        let batches = generate_batches(60, 20);
        assert_eq!(batches, vec![(0, 20), (20, 20), (40, 20)]);
    }

    #[test]
    fn test_estimate_search_size() {
        let params = Phase1Params::default();
        let size = estimate_search_size(&params, 50);
        assert_eq!(size, 10_000 * 50);
    }
}
