use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
};
mod node;
use node::Node;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Vector<const N: usize>(Vec<f32>);

impl<const N: usize> From<[f32; N]> for Vector<N> {
    fn from(values: [f32; N]) -> Self {
        Self(values.to_vec())
    }
}

impl<const N: usize> Hash for Vector<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, N * 4) };
        bytes.hash(state);
    }
}

impl<const N: usize> PartialEq for Vector<N> {
    // Eq when the underlying memory is the same. Very strict.
    fn eq(&self, other: &Self) -> bool {
        let bytes_left: &[u8] =
            unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, N * 4) };
        let bytes_right: &[u8] =
            unsafe { std::slice::from_raw_parts(other.0.as_ptr() as *const u8, N * 4) };
        bytes_left == bytes_right
    }
}

impl<const N: usize> Eq for Vector<N> {}

impl<const N: usize> Vector<N> {
    /// Try to create a Vector from a Vec<f32>.
    ///
    /// # Arguments
    ///
    /// * `values` - The vector to create the Vector from, must be of size N.
    ///
    /// # Returns
    ///
    /// * `Ok(Vector<N>)` - The Vector created from the Vec<f32>.
    /// * `Err(Vec<f32>)` - The Vec<f32> is not the correct size.
    pub fn try_from(values: Vec<f32>) -> Result<Self, Vec<f32>> {
        if values.len() != N {
            Err(values)
        } else {
            Ok(Self(values))
        }
    }

    pub fn slice(&self) -> &[f32; N] {
        // Convert the Vec<f32> slice to a fixed-size array reference
        // This is safe because we know the Vector always has exactly N elements
        unsafe { &*(self.0.as_ptr() as *const [f32; N]) }
    }

    pub fn subtract_from(&self, other: &Vector<N>) -> Vector<N> {
        let vals = self
            .slice()
            .iter()
            .zip(other.slice().iter())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        Vector(vals)
    }

    pub fn add(&self, vector: &Vector<N>) -> Vector<N> {
        let vals = self
            .slice()
            .iter()
            .zip(vector.slice().iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        Vector(vals)
    }

    pub fn avg(&self, vector: &Vector<N>) -> Vector<N> {
        let vals = self
            .slice()
            .iter()
            .zip(vector.slice().iter())
            .map(|(a, b)| (a + b) / 2.0)
            .collect::<Vec<_>>();
        Vector(vals)
    }

    pub fn dot_product(&self, vector: &Vector<N>) -> f32 {
        self.slice()
            .iter()
            .zip(vector.slice().iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn sq_euc_dist(&self, vector: &Vector<N>) -> f32 {
        self.slice()
            .iter()
            .zip(vector.slice().iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
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

pub struct ANNIndex<'a, const N: usize> {
    trees: Vec<Node<N>>,
    values: &'a [Vector<N>],
}

impl<'a, const N: usize> ANNIndex<'a, N> {
    /// Build the index for the given vectors and ids.
    ///
    /// # Arguments
    ///
    /// * `num_trees` - The number of trees to build, higher means more accurate but slower and larger memory usage.
    /// * `max_leaf_size` - The maximum number of vectors in a leaf node, lower means higher accuracy but slower search.
    /// * `vectors` - The vectors to build the index from.
    pub fn build<R: Rng>(
        num_trees: i32,
        max_leaf_size: i32,
        vectors: &'a [Vector<N>],
        rng: &mut R,
    ) -> Self {
        assert!(check_unique(vectors), "Vectors are not unique");

        let all_indexes: Vec<usize> = (0..vectors.len()).collect();

        let trees: Vec<_> = (0..num_trees)
            .map(|_| Node::build_tree(max_leaf_size, &all_indexes, vectors, rng))
            .collect();

        Self {
            trees,
            values: vectors,
        }
    }

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
    pub fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<(usize, f32)> {
        let mut candidates = HashSet::new();
        for tree in self.trees.iter() {
            tree.search(query, top_k, &mut candidates);
        }

        let mut results = candidates
            .into_iter()
            .map(|idx| (idx, self.values[idx].sq_euc_dist(query)))
            .collect::<Vec<_>>();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
            .into_iter()
            .take(top_k as usize)
            .map(|(idx, dis)| (idx, dis))
            .collect()
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
        let index = ANNIndex::build(1, 1, &vectors, &mut seed_rng);

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
        let index = ANNIndex::build(1, 2, &vectors, &mut seed_rng);

        // Query vectors itself should return exact matches
        for (i, vector) in vectors.iter().enumerate() {
            let results = index.search(vector, 2);
            assert_eq!(results.len(), 2);

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
        let index = ANNIndex::build(3, 2, &vectors, &mut seed_rng);

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
