use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::physics::{GeometryCandidate, SearchResult};

/// Wrapper for GeometryCandidate that implements Ord by score.
/// Used with BinaryHeap (max-heap) so the root holds the *worst* (highest-score)
/// candidate, allowing efficient eviction when capacity is exceeded.
#[derive(Clone, Debug)]
struct OrdCandidate(GeometryCandidate);

impl PartialEq for OrdCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.0.score == other.0.score
    }
}

impl Eq for OrdCandidate {}

impl PartialOrd for OrdCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // f64 total ordering: treat NaN as greater so it gets evicted first
        self.0
            .score
            .partial_cmp(&other.0.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// A bounded priority queue that keeps the top N candidates with the lowest scores.
pub struct ResultCollector {
    max_candidates: usize,
    // Max-heap: root is the worst (highest score) kept candidate,
    // so we can efficiently evict it when a better one arrives.
    heap: BinaryHeap<OrdCandidate>,
}

impl ResultCollector {
    pub fn new(max_candidates: usize) -> Self {
        Self {
            max_candidates,
            heap: BinaryHeap::with_capacity(max_candidates + 1),
        }
    }

    /// Add a candidate. If the collector is at capacity, the worst candidate is evicted.
    pub fn add(&mut self, candidate: GeometryCandidate) {
        self.heap.push(OrdCandidate(candidate));
        if self.heap.len() > self.max_candidates {
            self.heap.pop(); // remove worst (highest score = max-heap root)
        }
    }

    /// Return the best (lowest score) candidate, if any.
    pub fn best(&self) -> Option<&GeometryCandidate> {
        self.heap
            .iter()
            .min_by(|a, b| {
                a.0.score
                    .partial_cmp(&b.0.score)
                    .unwrap_or(Ordering::Equal)
            })
            .map(|r| &r.0)
    }

    /// Return the top N candidates sorted by score (best first).
    pub fn top_n(&self, n: usize) -> Vec<GeometryCandidate> {
        let mut items: Vec<GeometryCandidate> =
            self.heap.iter().map(|r| r.0.clone()).collect();
        items.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));
        items.truncate(n);
        items
    }

    /// Remove duplicate candidates: merge entries with the same mode pair
    /// encoding (p == p, q == q) and nearly identical rho (within 1e-8).
    /// Keeps only the best (lowest) score for each unique geometry.
    pub fn dedup(&mut self) {
        let mut items: Vec<GeometryCandidate> =
            self.heap.drain().map(|r| r.0).collect();
        items.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));

        let mut kept: Vec<GeometryCandidate> = Vec::new();
        for candidate in items {
            let dominated = kept.iter().any(|k| {
                k.p == candidate.p
                    && k.q == candidate.q
                    && (k.rho - candidate.rho).abs() < 1e-8
            });
            if !dominated {
                kept.push(candidate);
            }
        }

        // Re-insert into a fresh heap
        self.heap = BinaryHeap::with_capacity(self.max_candidates + 1);
        for c in kept.into_iter().take(self.max_candidates) {
            self.heap.push(OrdCandidate(c));
        }
    }

    /// Serialize all kept candidates to a JSON file.
    pub fn save_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let candidates = self.top_n(self.max_candidates);
        let json = serde_json::to_string_pretty(&candidates)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Format a u64 with comma-separated thousands.
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Print a single candidate to stdout.
pub fn print_candidate(c: &GeometryCandidate) {
    println!(
        "  ({},{}) rho={:.6} eps={:.2e} path={:.8} ratio={:.10} score={:.2e} [{}]",
        c.p, c.q, c.rho, c.epsilon, c.path_length, c.ratio, c.score, c.method
    );
}

/// Print a phase summary to stdout.
pub fn print_summary(result: &SearchResult) {
    println!("--- Phase {} Summary ---", result.phase);
    println!(
        "Evaluated: {} candidates in {:.2}s",
        format_number(result.total_evaluated),
        result.elapsed_secs
    );
    println!("Best score: {:.2e}", result.best_score);
    println!("Top candidates:");
    for c in &result.candidates {
        print_candidate(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(score: f64) -> GeometryCandidate {
        GeometryCandidate {
            rho: 0.5,
            p: 1,
            q: 1,
            epsilon: 0.0,
            path_length: 1.0,
            ratio: 206.0 + score,
            score,
            method: "test".to_string(),
        }
    }

    #[test]
    fn test_collector_keeps_best() {
        let mut rc = ResultCollector::new(3);
        for s in [5.0, 1.0, 3.0, 0.5, 4.0, 2.0] {
            rc.add(make_candidate(s));
        }
        let top = rc.top_n(3);
        assert_eq!(top.len(), 3);
        assert!((top[0].score - 0.5).abs() < 1e-15);
        assert!((top[1].score - 1.0).abs() < 1e-15);
        assert!((top[2].score - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_best() {
        let mut rc = ResultCollector::new(5);
        rc.add(make_candidate(3.0));
        rc.add(make_candidate(1.0));
        rc.add(make_candidate(2.0));
        assert!((rc.best().unwrap().score - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1234567), "1,234,567");
        assert_eq!(format_number(42), "42");
        assert_eq!(format_number(1000), "1,000");
    }
}
