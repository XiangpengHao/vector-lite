use crate::{Node, Vector};
use bincode::{Decode, Encode};
use rand::Rng;
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

pub enum ScoreMetric {
    Cosine,
    L2,
}

pub trait ANNIndexOwned<const N: usize> {
    /// Insert a vector into the index.
    fn insert(&mut self, vector: Vector<N>, id: String) {
        self.insert_with_rng(vector, id, &mut rand::rng());
    }

    /// Insert a vector into the index with a custom rng.
    fn insert_with_rng(&mut self, vector: Vector<N>, id: String, rng: &mut impl Rng);

    /// Delete a vector from the index by id.
    fn delete_by_id(&mut self, id: String);

    /// Get a vector from the index by id.
    fn get_by_id(&self, id: String) -> Option<&Vector<N>>;

    /// Search for the top_k nearest neighbors of the query vector.
    /// Returns a array of (id, score) pairs, higher score means closer.
    fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<(String, f32)> {
        self.search_with_metric(query, top_k, ScoreMetric::L2)
    }

    /// Search for the top_k nearest neighbors of the query vector.
    /// Returns a array of (id, score) pairs, higher score means closer.
    fn search_with_metric(
        &self,
        query: &Vector<N>,
        top_k: usize,
        metric: ScoreMetric,
    ) -> Vec<(String, f32)>;
}

#[derive(Encode, Decode)]
pub struct VectorLiteIndex<const N: usize> {
    trees: Vec<Node<N, Rc<String>>>,
    max_leaf_size: usize,
}

impl<const N: usize> VectorLiteIndex<N> {
    fn new(num_trees: usize, max_leaf_size: usize) -> Self {
        Self {
            trees: (0..num_trees).map(|_| Node::new_empty()).collect(),
            max_leaf_size,
        }
    }

    /// Search for the top_k nearest neighbors of the query vector.
    /// Returns a vector of ids, user may use the ids to compute the distance themselves.
    /// This is helpful when the vector is stored in a different place.
    pub fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<String> {
        let mut candidates = HashSet::new();
        for tree in &self.trees {
            tree.search(query, top_k, &mut candidates);
        }
        candidates
            .into_iter()
            .map(|id| id.as_ref().clone())
            .collect()
    }

    /// Serialize the index to a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let config = bincode::config::standard();
        bincode::encode_to_vec(self, config).unwrap()
    }

    /// Deserialize the index from a byte vector.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let config = bincode::config::standard();
        let (index, _) = bincode::decode_from_slice(bytes, config).unwrap();
        index
    }
}

/// A lightweight lsh-based ann index.
#[derive(Encode, Decode)]
pub struct VectorLite<const N: usize> {
    vectors: HashMap<Rc<String>, Vector<N>>,
    index: VectorLiteIndex<N>,
}

impl<const N: usize> VectorLite<N> {
    /// Create a new VectorLite index with the given number of trees and max leaf size.
    /// More trees means more accuracy but slower search and larger memory usage.
    /// Lower max_leaf_size means higher accuracy but larger memory usage.
    pub fn new(num_trees: usize, max_leaf_size: usize) -> Self {
        Self {
            vectors: HashMap::new(),
            index: VectorLiteIndex::new(num_trees, max_leaf_size),
        }
    }

    /// Get the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get the index.
    pub fn index(&self) -> &VectorLiteIndex<N> {
        &self.index
    }

    /// Serialize the index to a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let config = bincode::config::standard();
        bincode::encode_to_vec(self, config).unwrap()
    }

    /// Deserialize the index from a byte vector.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let config = bincode::config::standard();
        let (index, _) = bincode::decode_from_slice(bytes, config).unwrap();
        index
    }
}

impl<const N: usize> ANNIndexOwned<N> for VectorLite<N> {
    fn insert_with_rng(&mut self, vector: Vector<N>, id: String, rng: &mut impl Rng) {
        let id = Rc::new(id);
        self.vectors.insert(id.clone(), vector);
        let vector_fn = |id: &Rc<String>| &self.vectors[id];
        for tree in &mut self.index.trees {
            tree.insert(&vector_fn, id.clone(), rng, self.index.max_leaf_size);
        }
    }

    fn delete_by_id(&mut self, id: String) {
        let id = Rc::new(id);
        for tree in &mut self.index.trees {
            tree.delete(&self.vectors[&id], &id);
        }
        self.vectors.remove(&id);
    }

    fn get_by_id(&self, id: String) -> Option<&Vector<N>> {
        self.vectors.get(&id)
    }

    fn search_with_metric(
        &self,
        query: &Vector<N>,
        top_k: usize,
        metric: ScoreMetric,
    ) -> Vec<(String, f32)> {
        let candidates = self.index.search(query, top_k);

        match metric {
            ScoreMetric::L2 => {
                let mut results = candidates
                    .into_iter()
                    .map(|id| {
                        let dist = self.vectors[&id].sq_euc_dist(query);
                        (id, dist)
                    })
                    .collect::<Vec<_>>();
                results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                results.into_iter().take(top_k).collect()
            }
            ScoreMetric::Cosine => {
                let mut results = candidates
                    .into_iter()
                    .map(|id| {
                        let dist = self.vectors[&id].cosine_similarity(query);
                        (id, dist)
                    })
                    .collect::<Vec<_>>();
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                results.into_iter().take(top_k).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    // Helper function to create a deterministic RNG for testing
    fn create_test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_basic_operations() {
        let mut index = VectorLite::<3>::new(2, 2);
        let mut rng = create_test_rng();

        // Insert some vectors
        index.insert_with_rng(Vector::from([1.0, 0.0, 0.0]), "101".to_string(), &mut rng);
        index.insert_with_rng(Vector::from([0.0, 1.0, 0.0]), "102".to_string(), &mut rng);
        index.insert_with_rng(Vector::from([0.0, 0.0, 1.0]), "103".to_string(), &mut rng);
        index.insert_with_rng(Vector::from([1.0, 1.0, 0.0]), "104".to_string(), &mut rng);

        // Search for nearest neighbors
        let results = index.search(&Vector::from([0.9, 0.1, 0.0]), 2);

        // Verify correct results
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "101"); // First result should be ID 101 ([1.0, 0.0, 0.0])

        // Query close to second vector
        let results = index.search(&Vector::from([0.1, 0.9, 0.0]), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "102"); // Should be ID 102 ([0.0, 1.0, 0.0])
    }

    #[test]
    fn test_multi_tree_performance() {
        // Compare performance with different numbers of trees
        let mut single_tree = VectorLite::<2>::new(1, 2);
        let mut multi_tree = VectorLite::<2>::new(5, 2);

        let mut rng = create_test_rng();

        // Create a grid of vectors
        for x in 0..5 {
            for y in 0..5 {
                let id = x * 10 + y;
                let vector = Vector::from([x as f32, y as f32]);

                // Insert into both indexes
                single_tree.insert_with_rng(vector.clone(), id.to_string(), &mut rng);
                multi_tree.insert_with_rng(vector, id.to_string(), &mut rng);
            }
        }

        // Query in a spot where approximation might occur
        let query = Vector::from([2.3, 2.3]);

        let single_results = single_tree.search(&query, 5);
        let multi_results = multi_tree.search(&query, 5);

        // Multi-tree should find at least as many results as single tree
        assert!(multi_results.len() >= single_results.len());

        assert_eq!(multi_results[0].0, "22");

        // The results should be sorted by distance
        for i in 1..multi_results.len() {
            assert!(multi_results[i].1 >= multi_results[i - 1].1);
        }
    }

    #[test]
    fn test_deletion() {
        let mut index = VectorLite::<2>::new(3, 2);
        let mut rng = create_test_rng();

        // Insert vectors with sequential IDs
        for i in 0..10 {
            let x = i as f32;
            index.insert_with_rng(Vector::from([x, x]), i.to_string(), &mut rng);
        }

        // Search for a point and verify it exists
        let results = index.search(&Vector::from([5.0, 5.0]), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "5");

        // Delete that point
        index.delete_by_id("5".to_string());

        // Search again - should find a different point now
        let results = index.search(&Vector::from([5.0, 5.0]), 1);
        assert_eq!(results.len(), 1);
        assert_ne!(results[0].0, "5"); // Should not find the deleted point

        // The nearest should now be either 4 or 6
        assert!(results[0].0 == "4" || results[0].0 == "6");

        // Delete several points and verify none are found
        index.delete_by_id("4".to_string());
        index.delete_by_id("6".to_string());

        let results = index.search(&Vector::from([5.0, 5.0]), 3);
        for result in results {
            assert!(result.0 != "4" && result.0 != "5" && result.0 != "6");
        }
    }

    #[test]
    fn test_file_operations() {
        // Create a new index
        let mut index = VectorLite::<3>::new(2, 2);
        let mut rng = create_test_rng();

        // Insert some test vectors
        index.insert_with_rng(Vector::from([1.0, 0.0, 0.0]), "101".to_string(), &mut rng);
        index.insert_with_rng(Vector::from([0.0, 1.0, 0.0]), "102".to_string(), &mut rng);
        index.insert_with_rng(Vector::from([0.0, 0.0, 1.0]), "103".to_string(), &mut rng);

        let serialized = index.to_bytes();
        let loaded_index = VectorLite::<3>::from_bytes(&serialized);

        // Verify search results match
        let query = Vector::from([0.9, 0.1, 0.0]);
        let original_results = index.search(&query, 2);
        let loaded_results = loaded_index.search(&query, 2);

        assert_eq!(original_results.len(), loaded_results.len());
        for i in 0..original_results.len() {
            assert_eq!(original_results[i].0, loaded_results[i].0);
            assert!((original_results[i].1 - loaded_results[i].1).abs() < 1e-6);
        }
    }
}
