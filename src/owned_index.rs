use std::collections::HashSet;

use rand::Rng;

use crate::{Node, Vector};

pub trait ANNIndexOwned<const N: usize> {
    fn insert(&mut self, vector: Vector<N>, id: usize, rng: &mut impl Rng);
    fn delete(&mut self, id: usize);
    fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<(usize, f32)>;
}

pub struct LshOwned<const N: usize> {
    vectors: Vec<Vector<N>>,
    ids: Vec<usize>,
    trees: Vec<Node<N>>,
    max_leaf_size: usize,
}

impl<const N: usize> LshOwned<N> {
    pub fn new(num_trees: usize, max_leaf_size: usize) -> Self {
        Self {
            vectors: Vec::new(),
            ids: Vec::new(),
            trees: (0..num_trees).map(|_| Node::new_empty()).collect(),
            max_leaf_size,
        }
    }
}

impl<const N: usize> ANNIndexOwned<N> for LshOwned<N> {
    fn insert(&mut self, vector: Vector<N>, id: usize, rng: &mut impl Rng) {
        self.vectors.push(vector);
        self.ids.push(id);
        for tree in &mut self.trees {
            tree.insert(
                &self.vectors,
                self.vectors.len() as u32 - 1,
                rng,
                self.max_leaf_size,
            );
        }
    }

    fn delete(&mut self, id: usize) {
        for tree in &mut self.trees {
            tree.delete(&self.vectors, id as u32);
        }
        // TODO: also remove from vectors and ids.
    }

    fn search(&self, query: &Vector<N>, top_k: usize) -> Vec<(usize, f32)> {
        let mut candidates = HashSet::new();
        for tree in self.trees.iter() {
            tree.search(query, top_k, &mut candidates);
        }

        let mut results = candidates
            .into_iter()
            .map(|idx| (idx as usize, self.vectors[idx as usize].sq_euc_dist(query)))
            .collect::<Vec<_>>();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.into_iter().take(top_k).collect()
    }
}
