use crate::{Node, Vector};
use async_trait::async_trait;
use bincode::{Decode, Encode};
use std::collections::{HashMap, HashSet};

pub enum ScoreMetric {
    Cosine,
    L2,
}

#[async_trait]
pub trait ANNIndexOwned<const N: usize> {
    /// Insert a vector into the index.
    async fn insert(&mut self, vector: Vector<N>, id: String);

    /// Delete a vector from the index by id.
    /// Returns true if the vector was deleted, false if it was not found.
    async fn delete_by_id(&mut self, id: &str) -> bool;

    /// Get a vector from the index by id.
    async fn get_by_id(&self, id: &str) -> Option<&Vector<N>>;

    /// Search for the top_k nearest neighbors of the query vector.
    /// Returns a array of (id, score) pairs, higher score means closer.
    async fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<(String, f32)> {
        self.search_with_metric(query, top_k, ScoreMetric::L2).await
    }

    /// Search for the top_k nearest neighbors of the query vector.
    /// Returns a array of (id, score) pairs, higher score means closer.
    async fn search_with_metric(
        &self,
        query: &Vector<N>,
        top_k: usize,
        metric: ScoreMetric,
    ) -> Vec<(String, f32)>;
}

#[derive(Encode, Decode)]
pub struct VectorLiteIndex<const N: usize> {
    trees: Vec<Node<N, String>>,
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
    pub fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<&String> {
        let mut candidates = HashSet::new();
        for tree in &self.trees {
            tree.search(query, top_k, &mut candidates);
        }
        candidates.into_iter().collect()
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
    vectors: HashMap<String, Vector<N>>,
    index: VectorLiteIndex<N>,
}

impl<const N: usize> Default for VectorLite<N> {
    fn default() -> Self {
        Self::new(4, 30)
    }
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

#[async_trait]
impl<const N: usize> ANNIndexOwned<N> for VectorLite<N> {
    async fn insert(&mut self, vector: Vector<N>, id: String) {
        self.vectors.insert(id.clone(), vector);
        let vector_fn = |id: &String| &self.vectors[id];
        for tree in &mut self.index.trees {
            tree.insert(&vector_fn, id.clone(), self.index.max_leaf_size);
        }
    }

    async fn delete_by_id(&mut self, id: &str) -> bool {
        let Some(vector) = self.vectors.remove(id) else {
            return false;
        };
        let id = id.to_string();
        for tree in &mut self.index.trees {
            tree.delete(&vector, &id);
        }
        true
    }

    async fn get_by_id(&self, id: &str) -> Option<&Vector<N>> {
        self.vectors.get(id)
    }

    async fn search_with_metric(
        &self,
        query: &Vector<N>,
        top_k: usize,
        metric: ScoreMetric,
    ) -> Vec<(String, f32)> {
        let candidates = self.index.search(query, top_k);

        let results = match metric {
            ScoreMetric::L2 => {
                let mut results = candidates
                    .into_iter()
                    .map(|id| {
                        let dist = self.vectors[id].sq_euc_dist(query);
                        (id, dist)
                    })
                    .collect::<Vec<_>>();
                results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                results
            }
            ScoreMetric::Cosine => {
                let mut results = candidates
                    .into_iter()
                    .map(|id| {
                        let dist = self.vectors[id].cosine_similarity(query);
                        (id, dist)
                    })
                    .collect::<Vec<_>>();
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                results
            }
        };

        results
            .into_iter()
            .take(top_k)
            .map(|(id, dist)| (id.clone(), dist))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    macro_rules! cross_test {
        (fn $name:ident() $body:block) => {
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
            #[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
            async fn $name() $body
        };
    }

    cross_test!(
        fn test_basic_operations() {
            let mut index = VectorLite::<3>::new(2, 2);

            // Insert some vectors
            index
                .insert(Vector::from([1.0, 0.0, 0.0]), "101".to_string())
                .await;
            index
                .insert(Vector::from([0.0, 1.0, 0.0]), "102".to_string())
                .await;
            index
                .insert(Vector::from([0.0, 0.0, 1.0]), "103".to_string())
                .await;
            index
                .insert(Vector::from([1.0, 1.0, 0.0]), "104".to_string())
                .await;

            // Search for nearest neighbors
            let results = index.search(&Vector::from([0.9, 0.1, 0.0]), 2).await;

            // Verify correct results
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0, "101"); // First result should be ID 101 ([1.0, 0.0, 0.0])

            // Query close to second vector
            let results = index.search(&Vector::from([0.1, 0.9, 0.0]), 1).await;
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, "102"); // Should be ID 102 ([0.0, 1.0, 0.0])
        }
    );

    cross_test!(
        fn test_multi_tree_performance() {
            // Compare performance with different numbers of trees
            let mut single_tree = VectorLite::<2>::new(1, 2);
            let mut multi_tree = VectorLite::<2>::new(5, 2);

            // Create a grid of vectors
            for x in 0..5 {
                for y in 0..5 {
                    let id = x * 10 + y;
                    let vector = Vector::from([x as f32, y as f32]);

                    // Insert into both indexes
                    single_tree.insert(vector.clone(), id.to_string()).await;
                    multi_tree.insert(vector, id.to_string()).await;
                }
            }

            // Query in a spot where approximation might occur
            let query = Vector::from([2.3, 2.3]);

            let single_results = single_tree.search(&query, 5).await;
            let multi_results = multi_tree.search(&query, 5).await;

            // Multi-tree should find at least as many results as single tree
            assert!(multi_results.len() >= single_results.len());

            assert_eq!(multi_results[0].0, "22");

            // The results should be sorted by distance
            for i in 1..multi_results.len() {
                assert!(multi_results[i].1 >= multi_results[i - 1].1);
            }
        }
    );

    cross_test!(
        fn test_deletion() {
            let mut index = VectorLite::<2>::new(3, 2);

            // Insert vectors with sequential IDs
            for i in 0..10 {
                let x = i as f32;
                index.insert(Vector::from([x, x]), i.to_string()).await;
            }

            // Search for a point and verify it exists
            let results = index.search(&Vector::from([5.0, 5.0]), 1).await;
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, "5");

            // Delete that point
            index.delete_by_id(&"5".to_string()).await;

            // Search again - should find a different point now
            let results = index.search(&Vector::from([5.0, 5.0]), 1).await;
            assert_eq!(results.len(), 1);
            assert_ne!(results[0].0, "5"); // Should not find the deleted point

            // The nearest should now be either 4 or 6
            assert!(results[0].0 == "4" || results[0].0 == "6");

            // Delete several points and verify none are found
            index.delete_by_id(&"4".to_string()).await;
            index.delete_by_id(&"6".to_string()).await;

            let results = index.search(&Vector::from([5.0, 5.0]), 3).await;
            for result in results {
                assert!(result.0 != "4" && result.0 != "5" && result.0 != "6");
            }
        }
    );

    cross_test!(
        fn test_file_operations() {
            // Create a new index
            let mut index = VectorLite::<3>::new(2, 2);

            // Insert some test vectors
            index
                .insert(Vector::from([1.0, 0.0, 0.0]), "101".to_string())
                .await;
            index
                .insert(Vector::from([0.0, 1.0, 0.0]), "102".to_string())
                .await;
            index
                .insert(Vector::from([0.0, 0.0, 1.0]), "103".to_string())
                .await;

            let serialized = index.to_bytes();
            let loaded_index = VectorLite::<3>::from_bytes(&serialized);

            // Verify search results match
            let query = Vector::from([0.9, 0.1, 0.0]);
            let original_results = index.search(&query, 2).await;
            let loaded_results = loaded_index.search(&query, 2).await;

            assert_eq!(original_results.len(), loaded_results.len());
            for i in 0..original_results.len() {
                assert_eq!(original_results[i].0, loaded_results[i].0);
                assert!((original_results[i].1 - loaded_results[i].1).abs() < 1e-6);
            }
        }
    );

    cross_test!(
        fn test_deleting_nonexistent_id() {
            let mut index = VectorLite::<3>::new(2, 2);

            index
                .insert(Vector::from([1.0, 0.0, 0.0]), "101".to_string())
                .await;
            index
                .insert(Vector::from([0.0, 1.0, 0.0]), "102".to_string())
                .await;

            let result = index.delete_by_id(&"non_existent_id".to_string()).await;

            assert_eq!(result, false);

            assert_eq!(index.len(), 2);
            assert!(index.get_by_id(&"101".to_string()).await.is_some());
            assert!(index.get_by_id(&"102".to_string()).await.is_some());

            let result = index.delete_by_id(&"101".to_string()).await;
            assert_eq!(result, true);
            assert_eq!(index.len(), 1);
        }
    );
}
