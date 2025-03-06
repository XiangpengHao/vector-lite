use crate::{Node, Vector};
use bincode::{Decode, Encode};
use rand::Rng;
use std::collections::{HashMap, HashSet};

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
    fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<(String, f32)>;
}

/// A lightweight lsh-based ann index.
#[derive(Encode, Decode)]
pub struct VectorLite<const N: usize> {
    vectors: Vec<Vector<N>>,
    id_to_offset: HashMap<String, u32>,
    offset_to_id: HashMap<u32, String>,
    trees: Vec<Node<N>>,
    max_leaf_size: usize,
}

impl<const N: usize> VectorLite<N> {
    /// Create a new VectorLite index with the given number of trees and max leaf size.
    /// More trees means more accuracy but slower search and larger memory usage.
    /// Lower max_leaf_size means higher accuracy but larger memory usage.
    pub fn new(num_trees: usize, max_leaf_size: usize) -> Self {
        Self {
            vectors: Vec::new(),
            id_to_offset: HashMap::new(),
            offset_to_id: HashMap::new(),
            trees: (0..num_trees).map(|_| Node::new_empty()).collect(),
            max_leaf_size,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let config = bincode::config::standard();
        bincode::encode_to_vec(self, config).unwrap()
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let config = bincode::config::standard();
        let (index, _) = bincode::decode_from_slice(bytes, config).unwrap();
        index
    }
}

impl<const N: usize> ANNIndexOwned<N> for VectorLite<N> {
    fn insert_with_rng(&mut self, vector: Vector<N>, id: String, rng: &mut impl Rng) {
        self.vectors.push(vector);
        let offset = self.vectors.len() as u32 - 1;
        self.offset_to_id.insert(offset, id.clone());
        self.id_to_offset.insert(id, offset);
        let vector_fn = |idx: u32| &self.vectors[idx as usize];
        for tree in &mut self.trees {
            tree.insert(
                &vector_fn,
                self.vectors.len() as u32 - 1,
                rng,
                self.max_leaf_size,
            );
        }
    }

    fn delete_by_id(&mut self, id: String) {
        let offset = self.id_to_offset[&id];
        for tree in &mut self.trees {
            tree.delete(&self.vectors[offset as usize], offset);
        }
        self.id_to_offset.remove(&id);
        self.offset_to_id.remove(&offset);
        // TODO: also remove from vectors.
    }

    fn get_by_id(&self, id: String) -> Option<&Vector<N>> {
        self.id_to_offset
            .get(&id)
            .map(|offset| &self.vectors[*offset as usize])
    }

    fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<(String, f32)> {
        let mut candidates = HashSet::new();
        for tree in self.trees.iter() {
            tree.search(query, top_k, &mut candidates);
        }

        let mut results = candidates
            .into_iter()
            .map(|offset| {
                (
                    offset,
                    self.vectors[offset as usize].cosine_similarity(query),
                )
            })
            .collect::<Vec<_>>();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
            .into_iter()
            .take(top_k)
            .map(|(offset, dist)| (self.offset_to_id[&offset].clone(), dist))
            .collect()
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
