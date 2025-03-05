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

fn deduplicate<const N: usize>(vectors: &[Vector<N>], ids: &[u32]) -> (Vec<Vector<N>>, Vec<u32>) {
    debug_assert!(vectors.len() == ids.len());
    let mut hashes_seen = HashSet::new();
    let mut dedup_vectors = Vec::new();
    let mut dedup_ids = Vec::new();
    for (vector, id) in vectors.iter().zip(ids.iter()) {
        if !hashes_seen.contains(&vector) {
            hashes_seen.insert(vector);
            dedup_vectors.push(vector.clone());
            dedup_ids.push(*id);
        }
    }

    (dedup_vectors, dedup_ids)
}

pub struct ANNIndex<const N: usize> {
    trees: Vec<Node<N>>,
    ids: Vec<u32>,
    values: Vec<Vector<N>>,
}

impl<const N: usize> ANNIndex<N> {
    /// Build the index for the given vectors and ids.
    ///
    /// # Arguments
    ///
    /// * `num_trees` - The number of trees to build, higher means more accurate but slower and larger memory usage.
    /// * `max_leaf_size` - The maximum number of vectors in a leaf node, lower means higher accuracy but slower search.
    /// * `vectors` - The vectors to build the index from.
    /// * `vec_ids` - The ids of the vectors.
    pub fn build<R: Rng>(
        num_trees: i32,
        max_leaf_size: i32,
        vectors: &[Vector<N>],
        vec_ids: &[u32],
        rng: &mut R,
    ) -> Self {
        debug_assert!(vectors.len() == vec_ids.len());
        let (unique_vector, ids) = deduplicate(vectors, vec_ids);

        let all_indexes: Vec<usize> = (0..unique_vector.len()).collect();

        let trees: Vec<_> = (0..num_trees)
            .map(|_| Node::build_tree(max_leaf_size, &all_indexes, &unique_vector, rng))
            .collect();

        Self {
            trees,
            ids,
            values: unique_vector,
        }
    }

    /// Search for the top_k nearest neighbors of the query vector.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector to search for.
    /// * `top_k` - The number of nearest neighbors to return.
    pub fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<(u32, f32)> {
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
            .map(|(idx, dis)| (self.ids[idx], dis))
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

        let ids = vec![1, 2, 3, 4, 5, 6];

        // Build the index with 1 trees and leaf max_size of 1, this will result in exact matches
        let index = ANNIndex::build(1, 1, &vectors, &ids, &mut seed_rng);

        // Query vectors itself should return exact matches
        for (i, vector) in vectors.iter().enumerate() {
            let results = index.search(vector, 1);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, ids[i]);
        }

        // Query vectors with a small distance should return the closest vector
        for (i, vector) in vectors.iter().enumerate() {
            let query = vector.add(&Vector::from([0.1, 0.1, 0.1]));
            let results = index.search(&query, 2);
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0, ids[i]);
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

        let ids = vec![1, 2, 3, 4, 5, 6];

        // Build the index with 1 trees and leaf max_size of 1, this will result in exact matches
        let index = ANNIndex::build(1, 2, &vectors, &ids, &mut seed_rng);

        // Query vectors itself should return exact matches
        for (i, vector) in vectors.iter().enumerate() {
            let results = index.search(vector, 2);
            assert_eq!(results.len(), 2);

            let id_bucket = i / 2 * 2;
            assert_eq!(
                results[0].0 + results[1].0,
                ids[id_bucket] + ids[id_bucket + 1]
            );
        }
    }

    #[test]
    fn test_duplicate_vectors() {
        let mut seed_rng = StdRng::seed_from_u64(42);
        let vectors = vec![
            Vector::from([1.0, 1.0, 1.0]),
            Vector::from([1.0, 1.0, 1.0]), // duplicate of the first vector
            Vector::from([2.0, 2.0, 2.0]),
        ];
        let ids = vec![10, 20, 30];

        let index = ANNIndex::build(5, 1, &vectors, &ids, &mut seed_rng);

        // Query exactly at the duplicate value.
        let query = Vector::from([1.0, 1.0, 1.0]);
        let top_k = 3;
        let results = index.search(&query, top_k);

        // Due to deduplication, only 2 unique vectors should be indexed.
        assert_eq!(results.len(), 2);

        // The nearest must be the duplicate vector (first occurrence, id 10) with 0.0 distance.
        assert_eq!(results[0].0, 10);
        assert!(approx_eq(results[0].1, 0.0, 0.0001));

        // The second result should be the distinct vector (id 30), whose squared distance is:
        //   (1-2)²+(1-2)²+(1-2)² = 1+1+1 = 3.
        assert_eq!(results[1].0, 30);
        assert!(approx_eq(results[1].1, 3.0, 0.0001));
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
        let ids = vec![100, 200, 300, 400, 500];

        // Build the index with 3 trees and a max_size of 2.
        let mut seed_rng = StdRng::seed_from_u64(42);
        let index = ANNIndex::build(3, 2, &vectors, &ids, &mut seed_rng);

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
