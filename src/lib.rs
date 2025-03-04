use std::{
    cmp::min,
    collections::HashSet,
    hash::{Hash, Hasher},
};

use rand::{Rng, rngs::StdRng, seq::IndexedRandom};

#[derive(Debug, Clone)]
pub struct Vector<const N: usize>([f32; N]);

impl<const N: usize> Vector<N> {
    pub fn new(values: [f32; N]) -> Self {
        Self(values)
    }
}

impl<const N: usize> Hash for Vector<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let bits: &[u32; N] = unsafe { std::mem::transmute(&self.0) };
        bits.hash(state);
    }
}

impl<const N: usize> PartialEq for Vector<N> {
    // Eq when the underlying memory is the same. Very strict.
    fn eq(&self, other: &Self) -> bool {
        let self_bits: &[u32; N] = unsafe { std::mem::transmute(&self.0) };
        let other_bits: &[u32; N] = unsafe { std::mem::transmute(&other.0) };
        self_bits == other_bits
    }
}

impl<const N: usize> Eq for Vector<N> {}

impl<const N: usize> Vector<N> {
    pub fn subtract_from(&self, other: &Vector<N>) -> Vector<N> {
        let mut coords = [0.0; N];
        for (v, o) in coords.iter_mut().zip(self.0.iter().zip(other.0.iter())) {
            *v = o.1 - o.0;
        }
        Vector::new(coords)
    }

    pub fn add(&self, vector: &Vector<N>) -> Vector<N> {
        let mut coords = [0.0; N];
        for (v, o) in coords.iter_mut().zip(self.0.iter().zip(vector.0.iter())) {
            *v = o.1 + o.0;
        }
        Vector::new(coords)
    }

    pub fn avg(&self, vector: &Vector<N>) -> Vector<N> {
        let mut coords = [0.0; N];
        for (v, o) in coords.iter_mut().zip(self.0.iter().zip(vector.0.iter())) {
            *v = (o.1 + o.0) / 2.0;
        }
        Vector::new(coords)
    }

    pub fn dot_product(&self, vector: &Vector<N>) -> f32 {
        self.0.iter().zip(vector.0.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn sq_euc_dist(&self, vector: &Vector<N>) -> f32 {
        self.0
            .iter()
            .zip(vector.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    }
}

struct HyperPlane<const N: usize> {
    coefficients: Vector<N>,
    constant: f32,
}

impl<const N: usize> HyperPlane<N> {
    pub fn point_is_above(&self, point: &Vector<N>) -> bool {
        self.coefficients.dot_product(point) + self.constant >= 0.0
    }
}

enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode<N>>),
}
struct LeafNode<const N: usize>(Vec<usize>);

struct InnerNode<const N: usize> {
    hyperplane: HyperPlane<N>,
    above: Box<Node<N>>,
    below: Box<Node<N>>,
}

pub struct ANNIndex<const N: usize> {
    trees: Vec<Node<N>>,
    ids: Vec<i32>,
    values: Vec<Vector<N>>,
}

impl<const N: usize> ANNIndex<N> {
    fn build_hyperplane<R: Rng>(
        indexes: &[usize],
        all_vectors: &[Vector<N>],
        rng: &mut R,
    ) -> (HyperPlane<N>, Vec<usize>, Vec<usize>) {
        let mut sample_iter = indexes.choose_multiple(rng, 2);

        let p1 = &all_vectors[*sample_iter.next().unwrap()];
        let p2 = &all_vectors[*sample_iter.next().unwrap()];

        let coefficients = p1.subtract_from(&p2);
        let point_on_plane = p1.avg(&p2);
        let constant = -coefficients.dot_product(&point_on_plane);
        let hyperplane = HyperPlane {
            coefficients,
            constant,
        };

        let mut above = Vec::new();
        let mut below = Vec::new();
        for &id in indexes.iter() {
            if hyperplane.point_is_above(&all_vectors[id]) {
                above.push(id);
            } else {
                below.push(id);
            }
        }

        (hyperplane, above, below)
    }

    fn build_a_tree(
        max_size: i32,
        indexes: &[usize],
        all_vectors: &[Vector<N>],
        rng: &mut StdRng,
    ) -> Node<N> {
        if indexes.len() <= max_size as usize {
            return Node::Leaf(Box::new(LeafNode(indexes.to_owned())));
        }

        let (hyperplane, above, below) = Self::build_hyperplane(indexes, all_vectors, rng);
        let node_above = Self::build_a_tree(max_size, &above, all_vectors, rng);
        let node_below = Self::build_a_tree(max_size, &below, all_vectors, rng);

        Node::Inner(Box::new(InnerNode {
            hyperplane,
            above: Box::new(node_above),
            below: Box::new(node_below),
        }))
    }

    pub fn build_index(
        num_trees: i32,
        max_leaf_size: i32,
        vectors: &[Vector<N>],
        vec_ids: &[i32],
        rng: &mut StdRng,
    ) -> Self {
        debug_assert!(vectors.len() == vec_ids.len());
        let (unique_vector, ids) = deduplicate(vectors, vec_ids);

        let all_indexes: Vec<usize> = (0..unique_vector.len()).collect();

        let trees: Vec<_> = (0..num_trees)
            .map(|_| Self::build_a_tree(max_leaf_size, &all_indexes, &unique_vector, rng))
            .collect();

        Self {
            trees,
            ids,
            values: unique_vector,
        }
    }

    fn tree_result(
        query: &Vector<N>,
        n: i32,
        tree: &Node<N>,
        candidates: &mut HashSet<usize>,
    ) -> i32 {
        match tree {
            Node::Leaf(leaf) => {
                let leaf_values = &(leaf.0);
                let num_candidates_found = min(n as usize, leaf_values.len());
                for value in leaf_values.iter().take(num_candidates_found) {
                    candidates.insert(*value);
                }
                num_candidates_found as i32
            }
            Node::Inner(inner) => {
                let above = inner.hyperplane.point_is_above(query);
                let (main, backup) = match above {
                    true => (&inner.above, &inner.below),
                    false => (&inner.below, &inner.above),
                };

                match Self::tree_result(query, n, main, candidates) {
                    k if k < n => k + Self::tree_result(query, n - k, backup, candidates),
                    k => k,
                }
            }
        }
    }

    pub fn search(&self, query: &Vector<N>, top_k: i32) -> Vec<(i32, f32)> {
        let mut candidates = HashSet::new();
        for tree in self.trees.iter() {
            Self::tree_result(query, top_k, tree, &mut candidates);
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

fn deduplicate<const N: usize>(vectors: &[Vector<N>], ids: &[i32]) -> (Vec<Vector<N>>, Vec<i32>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    // A small helper for approximate float equality.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_basic_nearest_neighbor() {
        let mut seed_rng = rand::SeedableRng::seed_from_u64(42);
        let vectors = vec![
            Vector::new([10.0, 20.0, 30.0]),
            Vector::new([10.0, 30.0, 20.0]),
            Vector::new([20.0, 10.0, 30.0]),
            Vector::new([20.0, 30.0, 10.0]),
            Vector::new([30.0, 20.0, 10.0]),
            Vector::new([30.0, 10.0, 20.0]),
        ];

        let ids = vec![1, 2, 3, 4, 5, 6];

        // Build the index with 1 trees and leaf max_size of 1, this will result in exact matches
        let index = ANNIndex::build_index(1, 1, &vectors, &ids, &mut seed_rng);

        // Query vectors itself should return exact matches
        for (i, vector) in vectors.iter().enumerate() {
            let results = index.search(vector, 1);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, ids[i]);
        }

        // Query vectors with a small distance should return the closest vector
        for (i, vector) in vectors.iter().enumerate() {
            let query = vector.add(&Vector::new([0.1, 0.1, 0.1]));
            let results = index.search(&query, 2);
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0, ids[i]);
        }
    }

    #[test]
    fn test_top_2_nearest_neighbor() {
        let mut seed_rng = rand::SeedableRng::seed_from_u64(42);
        let vectors = vec![
            Vector::new([10.0, 20.0, 30.0]),
            Vector::new([10.0, 20.0, 30.1]),
            Vector::new([20.0, 30.0, 10.0]),
            Vector::new([20.0, 30.1, 10.0]),
            Vector::new([30.0, 20.0, 10.0]),
            Vector::new([30.1, 20.0, 10.0]),
        ];

        let ids = vec![1, 2, 3, 4, 5, 6];

        // Build the index with 1 trees and leaf max_size of 1, this will result in exact matches
        let index = ANNIndex::build_index(1, 2, &vectors, &ids, &mut seed_rng);

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
        let mut seed_rng = rand::SeedableRng::seed_from_u64(42);
        let vectors = vec![
            Vector::new([1.0, 1.0, 1.0]),
            Vector::new([1.0, 1.0, 1.0]), // duplicate of the first vector
            Vector::new([2.0, 2.0, 2.0]),
        ];
        let ids = vec![10, 20, 30];

        let index = ANNIndex::build_index(5, 1, &vectors, &ids, &mut seed_rng);

        // Query exactly at the duplicate value.
        let query = Vector::new([1.0, 1.0, 1.0]);
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
            Vector::new([0.0, 0.0, 0.0, 0.0]),
            Vector::new([1.0, 0.0, 0.0, 0.0]),
            Vector::new([0.0, 1.0, 0.0, 0.0]),
            Vector::new([0.0, 0.0, 1.0, 0.0]),
            Vector::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let ids = vec![100, 200, 300, 400, 500];

        // Build the index with 3 trees and a max_size of 2.
        let mut seed_rng = rand::SeedableRng::seed_from_u64(42);
        let index = ANNIndex::build_index(3, 2, &vectors, &ids, &mut seed_rng);

        // Query with a vector that lies equidistant from all the given vectors.
        let query = Vector::new([0.5, 0.5, 0.5, 0.5]);
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
