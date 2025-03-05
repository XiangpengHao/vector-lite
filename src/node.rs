use std::{cmp::min, collections::HashSet};

use rand::{Rng, seq::IndexedRandom};

use crate::Vector;

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

pub(crate) enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode<N>>),
}

impl<const N: usize> Node<N> {
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

    pub(crate) fn build_tree<R: Rng>(
        max_size: i32,
        indexes: &[u32],
        all_vectors: &[Vector<N>],
        rng: &mut R,
    ) -> Node<N> {
        if indexes.len() <= max_size as usize {
            return Node::Leaf(Box::new(LeafNode(indexes.to_owned())));
        }

        let (hyperplane, above, below) = build_hyperplane(indexes, all_vectors, rng);
        let node_above = Self::build_tree(max_size, &above, all_vectors, rng);
        let node_below = Self::build_tree(max_size, &below, all_vectors, rng);

        Node::Inner(Box::new(InnerNode {
            hyperplane,
            above: node_above,
            below: node_below,
        }))
    }

    pub(crate) fn search(
        &self,
        query: &Vector<N>,
        n: usize,
        candidates: &mut HashSet<u32>,
    ) -> usize {
        match self {
            Node::Leaf(leaf) => {
                let leaf_values = &(leaf.0);
                let num_candidates_found = min(n as usize, leaf_values.len());
                for value in leaf_values.iter().take(num_candidates_found) {
                    candidates.insert(*value);
                }
                num_candidates_found
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
pub(crate) struct LeafNode<const N: usize>(Vec<u32>);

impl<const N: usize> LeafNode<N> {
    pub(crate) fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.0.len() * std::mem::size_of::<u32>()
    }
}

pub(crate) struct InnerNode<const N: usize> {
    hyperplane: HyperPlane<N>,
    above: Node<N>,
    below: Node<N>,
}

impl<const N: usize> InnerNode<N> {
    pub(crate) fn memory_usage(&self) -> usize {
        self.hyperplane.memory_usage() + self.above.memory_usage() + self.below.memory_usage()
    }
}

fn build_hyperplane<R: Rng, const N: usize>(
    indexes: &[u32],
    all_vectors: &[Vector<N>],
    rng: &mut R,
) -> (HyperPlane<N>, Vec<u32>, Vec<u32>) {
    let mut sample_iter = indexes.choose_multiple(rng, 2);

    let p1 = &all_vectors[*sample_iter.next().unwrap() as usize];
    let p2 = &all_vectors[*sample_iter.next().unwrap() as usize];

    let coefficients = p1.subtract_from(p2);
    let point_on_plane = p1.avg(p2);
    let constant = -coefficients.dot_product(&point_on_plane);
    let hyperplane = HyperPlane {
        coefficients,
        constant,
    };

    let mut above = Vec::new();
    let mut below = Vec::new();
    for &id in indexes.iter() {
        if hyperplane.point_is_above(&all_vectors[id as usize]) {
            above.push(id);
        } else {
            below.push(id);
        }
    }

    (hyperplane, above, below)
}
