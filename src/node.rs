use bincode::{Decode, Encode};
use rand::{Rng, seq::IndexedRandom};
use std::collections::HashSet;

use crate::{Vector, VectorKey};

#[derive(Encode, Decode)]
struct HyperPlane<const N: usize> {
    coefficients: Vector<N>,
    constant: f32,
}

impl<const N: usize> HyperPlane<N> {
    pub fn point_is_above(&self, point: &Vector<N>) -> bool {
        self.coefficients.dot_product(point) + self.constant >= 0.0
    }

    pub(crate) fn memory_usage(&self) -> usize {
        std::mem::size_of::<f32>() + self.coefficients.memory_usage()
    }
}

#[derive(Encode, Decode)]
pub(crate) enum Node<const N: usize, K: VectorKey> {
    Inner(Box<InnerNode<N, K>>),
    Leaf(Box<LeafNode<N, K>>),
}

impl<const N: usize, K: VectorKey> Node<N, K> {
    pub(crate) fn memory_usage(&self) -> usize {
        let self_size = std::mem::size_of::<Self>();
        match self {
            Node::Inner(inner) => self_size + inner.memory_usage(),
            Node::Leaf(leaf) => self_size + leaf.memory_usage(),
        }
    }

    pub(crate) fn inner_node_count(&self) -> usize {
        match self {
            Node::Inner(inner) => {
                1 + inner.above.inner_node_count() + inner.below.inner_node_count()
            }
            Node::Leaf(_) => 0,
        }
    }

    pub(crate) fn new_empty() -> Self {
        Node::Leaf(Box::new(LeafNode(Vec::new())))
    }

    pub(crate) fn build_tree<'a, R: Rng>(
        max_leaf_size: usize,
        vector_ids: &[K],
        vector_accessor: &impl Fn(&K) -> &'a Vector<N>,
        rng: &mut R,
    ) -> Node<N, K> {
        if vector_ids.len() <= max_leaf_size {
            return Node::Leaf(Box::new(LeafNode(vector_ids.to_owned())));
        }

        let (hyperplane, above, below) = build_hyperplane(vector_ids, vector_accessor, rng);
        let node_above = Self::build_tree(max_leaf_size, &above, vector_accessor, rng);
        let node_below = Self::build_tree(max_leaf_size, &below, vector_accessor, rng);

        Node::Inner(Box::new(InnerNode {
            hyperplane,
            above: node_above,
            below: node_below,
        }))
    }

    pub(crate) fn insert<'a>(
        &mut self,
        vector_accessor: &impl Fn(&K) -> &'a Vector<N>,
        vector_id: K,
        rng: &mut impl Rng,
        max_leaf_size: usize,
    ) {
        match self {
            Node::Leaf(leaf) => {
                leaf.0.push(vector_id);
                if leaf.0.len() > max_leaf_size {
                    let (hyperplane, above, below) =
                        build_hyperplane(&leaf.0, vector_accessor, rng);
                    let node_above = Self::build_tree(max_leaf_size, &above, vector_accessor, rng);
                    let node_below = Self::build_tree(max_leaf_size, &below, vector_accessor, rng);
                    *self = Node::Inner(Box::new(InnerNode {
                        hyperplane,
                        above: node_above,
                        below: node_below,
                    }));
                }
            }
            Node::Inner(inner) => {
                let is_above = inner.hyperplane.point_is_above(vector_accessor(&vector_id));

                match is_above {
                    true => inner
                        .above
                        .insert(vector_accessor, vector_id, rng, max_leaf_size),
                    false => inner
                        .below
                        .insert(vector_accessor, vector_id, rng, max_leaf_size),
                }
            }
        }
    }

    /// Returns true if the node needs to be merged with the parent
    pub(crate) fn delete(&mut self, vector: &Vector<N>, vector_id: &K) -> bool {
        match self {
            Node::Leaf(leaf) => {
                leaf.0.retain(|id| id != vector_id);
                leaf.0.is_empty()
            }
            Node::Inner(inner) => {
                let is_above = inner.hyperplane.point_is_above(vector);

                match is_above {
                    true => {
                        let needs_merge = inner.above.delete(vector, vector_id);
                        if needs_merge {
                            *self = std::mem::replace(
                                &mut inner.below,
                                Node::Leaf(Box::new(LeafNode(Vec::new()))),
                            );
                        }
                        false
                    }
                    false => {
                        let needs_merge = inner.below.delete(vector, vector_id);
                        if needs_merge {
                            *self = std::mem::replace(
                                &mut inner.above,
                                Node::Leaf(Box::new(LeafNode(Vec::new()))),
                            );
                        }
                        false
                    }
                }
            }
        }
    }

    pub(crate) fn search<'a, 'b: 'a>(
        &'b self,
        query: &Vector<N>,
        n: usize,
        candidates: &'a mut HashSet<&'b K>,
    ) -> usize {
        match self {
            Node::Leaf(leaf) => {
                for value in leaf.0.iter() {
                    candidates.insert(value);
                }
                leaf.0.len()
            }
            Node::Inner(inner) => {
                let above = inner.hyperplane.point_is_above(query);
                let (main, backup) = match above {
                    true => (&inner.above, &inner.below),
                    false => (&inner.below, &inner.above),
                };

                match main.search(query, n, candidates) {
                    k if k < n => k + backup.search(query, n - k, candidates),
                    k => k,
                }
            }
        }
    }
}
#[derive(Encode, Decode)]
pub(crate) struct LeafNode<const N: usize, K: VectorKey>(Vec<K>); // vector_id

impl<const N: usize, K: VectorKey> LeafNode<N, K> {
    pub(crate) fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.0.capacity()
    }
}

#[derive(Encode, Decode)]
pub(crate) struct InnerNode<const N: usize, K: VectorKey> {
    hyperplane: HyperPlane<N>,
    above: Node<N, K>,
    below: Node<N, K>,
}

impl<const N: usize, K: VectorKey> InnerNode<N, K> {
    pub(crate) fn memory_usage(&self) -> usize {
        self.hyperplane.memory_usage() + self.above.memory_usage() + self.below.memory_usage()
    }
}

fn build_hyperplane<'a, R: Rng, const N: usize, K: VectorKey>(
    all_ids: &[K],
    vector_accessor: &impl Fn(&K) -> &'a Vector<N>,
    rng: &mut R,
) -> (HyperPlane<N>, Vec<K>, Vec<K>) {
    let mut sample_iter = all_ids.choose_multiple(rng, 2);

    let p1 = vector_accessor(sample_iter.next().unwrap());
    let p2 = vector_accessor(sample_iter.next().unwrap());

    let coefficients = p1.subtract_from(p2);
    let point_on_plane = p1.avg(p2);
    let constant = -coefficients.dot_product(&point_on_plane);
    let hyperplane = HyperPlane {
        coefficients,
        constant,
    };

    let mut above = Vec::new();
    let mut below = Vec::new();
    for id in all_ids.iter() {
        if hyperplane.point_is_above(vector_accessor(id)) {
            above.push(id.clone());
        } else {
            below.push(id.clone());
        }
    }

    (hyperplane, above, below)
}

#[cfg(test)]
mod tests {
    use crate::{Node, Vector};
    use rand::{SeedableRng, rngs::StdRng};
    use std::collections::HashSet;

    // Helper function to create a deterministic RNG for testing
    fn create_test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    macro_rules! cross_test {
        (fn $name:ident() $body:block) => {
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
            #[cfg_attr(not(target_arch = "wasm32"), test)]
            fn $name() $body
        };
    }

    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    cross_test!(
        fn test_insert_and_search_basic() {
            // Create a set of 3D vectors
            let vectors = vec![
                Vector::from([1.0, 0.0, 0.0]),
                Vector::from([0.0, 1.0, 0.0]),
                Vector::from([0.0, 0.0, 1.0]),
                Vector::from([1.0, 1.0, 0.0]),
                Vector::from([0.0, 1.0, 1.0]),
                Vector::from([1.0, 0.0, 1.0]),
            ];
            let vector_fn = |id: &String| &vectors[id.parse::<usize>().unwrap()];

            let mut rng = create_test_rng();
            let max_leaf_size = 1;

            // Create initial tree with first 2 vectors
            let ids = vec!["0".to_string(), "1".to_string()];
            let mut node = Node::build_tree(max_leaf_size, &ids, &vector_fn, &mut rng);

            // Insert remaining vectors one by one
            for i in 2..vectors.len() {
                node.insert(&vector_fn, i.to_string(), &mut rng, max_leaf_size);
            }

            // Test search with a specific query that should find exact matches
            let query = Vector::from([1.0, 0.0, 0.0]); // Identical to vectors[0]
            let mut candidates = HashSet::new();
            let found = node.search(&query, 1, &mut candidates);

            // Should find exactly 1 candidate
            assert_eq!(found, 1);
            assert_eq!(candidates.len(), 1);

            // The nearest vector should be index 0
            assert!(candidates.contains(&"0".to_string()));

            // Try with a higher top_k value
            candidates.clear();
            let found = node.search(&query, 3, &mut candidates);

            // Should find 3 candidates
            assert_eq!(found, 3);
            assert_eq!(candidates.len(), 3);

            // Verify we can find an injected vector
            let query = Vector::from([0.0, 1.0, 1.0]); // Identical to vectors[4]
            candidates.clear();
            node.search(&query, 1, &mut candidates);
            assert!(candidates.contains(&"4".to_string()));
        }
    );

    cross_test!(
        fn test_search_approximate_nearest() {
            // Create a set of 4D vectors
            let vectors = vec![
                Vector::from([1.0, 2.0, 3.0, 4.0]),
                Vector::from([1.1, 2.1, 3.1, 4.1]), // Close to vector 0
                Vector::from([5.0, 6.0, 7.0, 8.0]),
                Vector::from([5.1, 6.1, 7.1, 8.1]), // Close to vector 2
                Vector::from([9.0, 10.0, 11.0, 12.0]),
                Vector::from([9.05, 10.05, 11.05, 12.05]), // Very close to vector 4
            ];
            let vector_fn = |id: &String| &vectors[id.parse::<usize>().unwrap()];
            let mut rng = create_test_rng();
            let max_leaf_size = 2;

            // Initialize the node with all vectors at once
            let ids: Vec<String> = (0..vectors.len() as u32).map(|i| i.to_string()).collect();
            let node = Node::build_tree(max_leaf_size, &ids, &vector_fn, &mut rng);

            // Query with a vector close to vectors[0] and vectors[1]
            let query = Vector::from([1.05, 2.05, 3.05, 4.05]);
            let mut candidates = HashSet::new();
            node.search(&query, 2, &mut candidates);

            // Should find the two nearest neighbors (indexes 0 and 1)
            assert!(candidates.contains(&"0".to_string()));
            assert!(candidates.contains(&"1".to_string()));

            // Query with a vector close to vectors[4] and vectors[5]
            let query = Vector::from([9.02, 10.02, 11.02, 12.02]);
            candidates.clear();
            node.search(&query, 2, &mut candidates);

            // Should find the two nearest neighbors (indexes 4 and 5)
            assert!(candidates.contains(&"4".to_string()));
            assert!(candidates.contains(&"5".to_string()));

            // Test search efficiency - should prioritize exploring relevant branches
            let query = Vector::from([5.05, 6.05, 7.05, 8.05]);
            candidates.clear();
            let found = node.search(&query, 2, &mut candidates);

            // Should find exactly 2 candidates
            assert_eq!(found, 2);
            assert_eq!(candidates.len(), 2);

            // The nearest vectors should be index 2 and 3
            assert!(candidates.contains(&"2".to_string()));
            assert!(candidates.contains(&"3".to_string()));
        }
    );

    cross_test!(
        fn test_incremental_insertion() {
            // Create a set of 2D vectors
            let vectors = vec![
                Vector::from([0.0, 0.0]),
                Vector::from([0.0, 1.0]),
                Vector::from([1.0, 0.0]),
                Vector::from([1.0, 1.0]),
                Vector::from([0.5, 0.5]), // Center point
                Vector::from([0.2, 0.2]), // Close to origin
                Vector::from([0.8, 0.8]), // Close to (1,1)
            ];
            let vector_fn = |id: &String| &vectors[id.parse::<usize>().unwrap()];
            let mut rng = create_test_rng();
            let max_leaf_size = 1; // Force splits to happen frequently

            // Start with an empty tree (just one vector)
            let ids = vec!["0".to_string()];
            let mut node = Node::build_tree(max_leaf_size, &ids, &vector_fn, &mut rng);

            // Insert vectors incrementally and test the structure after each insertion
            node.insert(&vector_fn, "1".to_string(), &mut rng, max_leaf_size);
            assert_eq!(node.inner_node_count(), 1); // Should have created an inner node

            // Insert more vectors
            for i in 2..5 {
                node.insert(&vector_fn, i.to_string(), &mut rng, max_leaf_size);
            }

            // Test if we can find vectors accurately after multiple insertions
            let query = Vector::from([0.0, 0.0]); // Origin
            let mut candidates = HashSet::new();
            node.search(&query, 2, &mut candidates);

            // Should find the closest vectors to origin (index 0 and likely 5)
            assert!(candidates.contains(&"0".to_string()));

            // Query near (1,1)
            let query = Vector::from([0.9, 0.9]);

            let mut candidates = HashSet::new();
            node.search(&query, 1, &mut candidates);

            // Should find vector closest to (1,1) which is index 3
            assert!(candidates.contains(&"3".to_string()));

            let mut candidates = HashSet::new();
            // Insert final vectors
            for i in 5..vectors.len() {
                node.insert(&vector_fn, i.to_string(), &mut rng, max_leaf_size);
            }

            // Test again with all vectors inserted
            let query = Vector::from([0.5, 0.5]); // Center
            candidates.clear();
            node.search(&query, 1, &mut candidates);

            // Should find the center point (index 4)
            assert!(candidates.contains(&"4".to_string()));
        }
    );

    cross_test!(
        fn test_delete_operations() {
            // Test 1: Basic leaf node deletion
            {
                let vectors = vec![Vector::from([1.0, 0.0]), Vector::from([0.0, 1.0])];
                let vector_fn = |id: &String| &vectors[id.parse::<usize>().unwrap()];

                let mut rng = create_test_rng();
                let max_leaf_size = 2;

                let ids = vec!["0".to_string(), "1".to_string()];
                let mut node = Node::build_tree(max_leaf_size, &ids, &vector_fn, &mut rng);

                let result = node.delete(&vectors[0], &"0".to_string());

                assert_eq!(result, false);

                let result = node.delete(&vectors[1], &"1".to_string());

                assert_eq!(result, true);
            }

            // Test 2: Node collapse when deleting from inner node
            {
                let vectors = vec![
                    Vector::from([0.0, 0.0]),   // below
                    Vector::from([10.0, 10.0]), // above
                ];

                let mut rng = create_test_rng();
                let max_leaf_size = 1; // Force smaller leaf nodes

                let ids: Vec<String> = (0..vectors.len() as u32).map(|i| i.to_string()).collect();
                let vector_fn = |id: &String| &vectors[id.parse::<usize>().unwrap()];
                let mut node = Node::build_tree(max_leaf_size, &ids, &vector_fn, &mut rng);

                match &node {
                    Node::Inner(_) => {}
                    _ => panic!("Expected an inner node"),
                }

                let result = node.delete(&vectors[1], &"1".to_string()); // Delete the "above" vector

                assert_eq!(result, false);

                match &node {
                    Node::Leaf(leaf) => {
                        assert_eq!(leaf.0.len(), 1);
                        assert_eq!(leaf.0[0], "0".to_string());
                    }
                    _ => panic!("Expected node to collapse to a leaf"),
                }
            }

            // Test 3: Cascading collapse in multi-level tree
            {
                let vectors = vec![
                    Vector::from([0.0, 0.0]),   // bottom left
                    Vector::from([0.0, 10.0]),  // top left
                    Vector::from([10.0, 0.0]),  // bottom right
                    Vector::from([10.0, 10.0]), // top right
                ];

                let mut rng = create_test_rng();
                let max_leaf_size = 1; // Force splits for structure

                let ids: Vec<String> = (0..vectors.len() as u32).map(|i| i.to_string()).collect();
                let vector_fn = |id: &String| &vectors[id.parse::<usize>().unwrap()];
                let mut node = Node::build_tree(max_leaf_size, &ids, &vector_fn, &mut rng);

                let initial_inner_count = node.inner_node_count();
                assert!(initial_inner_count > 0);

                node.delete(&vectors[2], &"2".to_string()); // Delete a vector from right side
                node.delete(&vectors[3], &"3".to_string()); // Delete the other vector from right side

                let after_delete_count = node.inner_node_count();
                assert!(after_delete_count < initial_inner_count);

                node.delete(&vectors[0], &"0".to_string());
                let final_result = node.delete(&vectors[1], &"1".to_string());

                assert_eq!(final_result, true);
            }
        }
    );
}
