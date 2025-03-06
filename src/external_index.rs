use std::collections::HashSet;

use rand::Rng;

use crate::{Node, Vector};

/// A simple LSH-based ANNIndex implementation.
pub struct LshExternal<'a, const N: usize> {
    trees: Vec<Node<N, u32>>,
    values: &'a [Vector<N>],
}

impl<const N: usize> LshExternal<'_, N> {
    /// Count the number of inner nodes in the index.
    pub fn inner_node_count(&self) -> usize {
        self.trees
            .iter()
            .map(|tree| tree.inner_node_count())
            .sum::<usize>()
    }
}

fn check_unique<const N: usize>(vectors: &[Vector<N>]) -> bool {
    let mut hashes_seen = HashSet::new();
    for vector in vectors.iter() {
        if !hashes_seen.insert(vector) {
            return false;
        }
    }
    true
}

impl<'a, const N: usize> ANNIndexExternal<'a, N> for LshExternal<'a, N> {
    type Index = Self;

    fn build<R: Rng>(
        num_trees: usize,
        max_leaf_size: usize,
        vectors: &'a [Vector<N>],
        rng: &mut R,
    ) -> Result<Self::Index, &'static str> {
        if !check_unique(vectors) {
            return Err("Vectors are not unique");
        }
        if vectors.len() > u32::MAX as usize {
            return Err("Number of vectors exceeds u32::MAX");
        }

        let all_indexes: Vec<u32> = (0..vectors.len() as u32).collect();
        let vector_fn = |idx: &u32| &vectors[*idx as usize];
        let trees: Vec<_> = (0..num_trees)
            .map(|_| Node::build_tree(max_leaf_size, &all_indexes, &vector_fn, rng))
            .collect();

        Ok(Self {
            trees,
            values: vectors,
        })
    }

    fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<(usize, f32)> {
        let mut candidates = HashSet::new();
        for tree in self.trees.iter() {
            tree.search(query, top_k, &mut candidates);
        }

        let mut results = candidates
            .into_iter()
            .map(|idx| (*idx as usize, self.values[*idx as usize].sq_euc_dist(query)))
            .collect::<Vec<_>>();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.into_iter().take(top_k).collect()
    }

    /// Get the memory usage of the index.
    fn memory_usage(&self) -> usize {
        let tree_size = self
            .trees
            .iter()
            .map(|tree| tree.memory_usage())
            .sum::<usize>();
        std::mem::size_of::<Self>() + tree_size
    }
}

/// A trait for ANNIndex that references external data.
/// The index is read-only once built.
pub trait ANNIndexExternal<'a, const N: usize> {
    type Index;

    /// Build the index for the given vectors and ids.
    ///
    /// # Arguments
    ///
    /// * `num_trees` - The number of trees to build, higher means more accurate but slower and larger memory usage.
    /// * `max_leaf_size` - The maximum number of vectors in a leaf node, lower means higher accuracy but slower search.
    /// * `vectors` - The vectors to build the index from.
    fn build<R: Rng>(
        num_trees: usize,
        max_leaf_size: usize,
        vectors: &'a [Vector<N>],
        rng: &mut R,
    ) -> Result<Self::Index, &'static str>;

    /// Search for the top_k nearest neighbors of the query vector.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector to search for.
    /// * `top_k` - The number of nearest neighbors to return.
    ///
    /// # Returns
    ///
    /// * `Vec<(usize, f32)>` - The top_k nearest (index, distance) of the query vector.
    fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<(usize, f32)>;

    /// Get the memory usage of the index.
    fn memory_usage(&self) -> usize;
}

/// A linear search implementation of the ANNIndex trait.
/// This performs no indexing and simply scans the entire dataset for each query.
/// It's useful as a baseline comparison and for small datasets.
pub struct LinearSearchExternal<'a, const N: usize> {
    values: &'a [Vector<N>],
}

impl<'a, const N: usize> ANNIndexExternal<'a, N> for LinearSearchExternal<'a, N> {
    type Index = Self;

    fn build<R: Rng>(
        _num_trees: usize,
        _max_leaf_size: usize,
        vectors: &'a [Vector<N>],
        _rng: &mut R,
    ) -> Result<Self::Index, &'static str> {
        if vectors.is_empty() {
            return Err("Cannot build index with empty vector set");
        }

        Ok(Self { values: vectors })
    }

    fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<(usize, f32)> {
        if top_k == 0 || self.values.is_empty() {
            return Vec::new();
        }

        // Use a custom struct to hold distance and index in a way that can be compared
        #[derive(PartialEq)]
        struct Entry(f32, usize);

        #[allow(clippy::non_canonical_partial_ord_impl)]
        impl PartialOrd for Entry {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }

        impl Eq for Entry {}

        impl Ord for Entry {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        // Use a max-heap to track only top_k elements
        let mut heap = std::collections::BinaryHeap::with_capacity(top_k);

        // Process each vector
        for (idx, vec) in self.values.iter().enumerate() {
            let dist = vec.sq_euc_dist(query);

            if heap.len() < top_k {
                // If we haven't reached capacity, add to the heap
                heap.push(Entry(dist, idx));
            } else if let Some(max_entry) = heap.peek() {
                // If current distance is smaller than the largest in our heap
                if dist < max_entry.0 {
                    // Remove the largest distance and add the new one
                    heap.pop();
                    heap.push(Entry(dist, idx));
                }
            }
        }

        // Convert heap back to vector of (idx, dist) pairs
        let mut result = Vec::with_capacity(heap.len());
        while let Some(Entry(dist, idx)) = heap.pop() {
            result.push((idx, dist));
        }

        // Reverse to get results in ascending order by distance
        result.reverse();
        result
    }

    fn memory_usage(&self) -> usize {
        // Only count the struct itself, not the referenced vectors
        std::mem::size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    // A small helper for approximate float equality.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_basic_nearest_neighbor() {
        let mut seed_rng = StdRng::seed_from_u64(42);
        let vectors = vec![
            Vector::from([10.0, 20.0, 30.0]),
            Vector::from([10.0, 30.0, 20.0]),
            Vector::from([20.0, 10.0, 30.0]),
            Vector::from([20.0, 30.0, 10.0]),
            Vector::from([30.0, 20.0, 10.0]),
            Vector::from([30.0, 10.0, 20.0]),
        ];

        // Build the index with 1 trees and leaf max_size of 1, this will result in exact matches
        let index = LshExternal::build(1, 1, &vectors, &mut seed_rng).unwrap();

        // Query vectors itself should return exact matches
        for (i, vector) in vectors.iter().enumerate() {
            let results = index.search(vector, 1);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, i);
        }

        // Query vectors with a small distance should return the closest vector
        for (i, vector) in vectors.iter().enumerate() {
            let query = vector.add(&Vector::from([0.1, 0.1, 0.1]));
            let results = index.search(&query, 2);
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0, i);
        }

        let index = LinearSearchExternal::build(1, 1, &vectors, &mut seed_rng).unwrap();
        for (i, vector) in vectors.iter().enumerate() {
            let results = index.search(vector, 1);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, i);
        }
        for (i, vector) in vectors.iter().enumerate() {
            let query = vector.add(&Vector::from([0.1, 0.1, 0.1]));
            let results = index.search(&query, 2);

            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0, i);
        }
    }

    #[test]
    fn test_top_2_nearest_neighbor() {
        let mut seed_rng = StdRng::seed_from_u64(42);
        let vectors = vec![
            Vector::from([10.0, 20.0, 30.0]),
            Vector::from([10.0, 20.0, 30.1]),
            Vector::from([20.0, 30.0, 10.0]),
            Vector::from([20.0, 30.1, 10.0]),
            Vector::from([30.0, 20.0, 10.0]),
            Vector::from([30.1, 20.0, 10.0]),
        ];

        // Build the index with 1 trees and leaf max_size of 1, this will result in exact matches
        let index = LshExternal::build(1, 2, &vectors, &mut seed_rng).unwrap();

        // Query vectors itself should return exact matches
        for (i, vector) in vectors.iter().enumerate() {
            let results = index.search(vector, 2);
            assert_eq!(results.len(), 2);

            let id_bucket = i / 2 * 2;
            assert_eq!(results[0].0 + results[1].0, id_bucket + id_bucket + 1);
        }

        let index = LinearSearchExternal::build(1, 2, &vectors, &mut seed_rng).unwrap();
        for (i, vector) in vectors.iter().enumerate() {
            let results = index.search(vector, 2);
            let id_bucket = i / 2 * 2;
            assert_eq!(results[0].0 + results[1].0, id_bucket + id_bucket + 1);
        }
    }

    /// Test 3: High-dimensional vectors with a top_k value exceeding the number of unique vectors.
    /// This test uses 4D vectors and checks that:
    /// - All available unique vectors are returned (when top_k is too high).
    /// - The computed squared distances are as expected.
    #[test]
    fn test_top_k_exceeds_total_and_high_dim() {
        // Create five 4D vectors.
        let vectors = vec![
            Vector::from([0.0, 0.0, 0.0, 0.0]),
            Vector::from([1.0, 0.0, 0.0, 0.0]),
            Vector::from([0.0, 1.0, 0.0, 0.0]),
            Vector::from([0.0, 0.0, 1.0, 0.0]),
            Vector::from([0.0, 0.0, 0.0, 1.0]),
        ];

        // Build the index with 3 trees and a max_size of 2.
        let mut seed_rng = StdRng::seed_from_u64(42);
        let index = LshExternal::build(3, 2, &vectors, &mut seed_rng).unwrap();

        // Query with a vector that lies equidistant from all the given vectors.
        let query = Vector::from([0.5, 0.5, 0.5, 0.5]);
        let top_k = 10; // Request more neighbors than there are unique vectors.
        let results = index.search(&query, top_k);

        // Since there are only 5 unique vectors, we expect 5 results.
        assert_eq!(results.len(), 5);

        // The squared distance for each vector should be exactly 1.0:
        // For example, for v0: (0.5-0.0)² * 4 = 0.25 * 4 = 1.0.
        // For the others, the differences still sum to 1.0.
        for &(_, dist) in results.iter() {
            assert!(approx_eq(dist, 1.0, 0.0001));
        }
    }
}
