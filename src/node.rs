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
}

pub(crate) enum Node<const N: usize> {
    Inner(Box<InnerNode<N>>),
    Leaf(Box<LeafNode<N>>),
}

impl<const N: usize> Node<N> {
    pub(crate) fn build_tree<R: Rng>(
        max_size: i32,
        indexes: &[usize],
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
        candidates: &mut HashSet<usize>,
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
pub(crate) struct LeafNode<const N: usize>(Vec<usize>);

pub(crate) struct InnerNode<const N: usize> {
    hyperplane: HyperPlane<N>,
    above: Node<N>,
    below: Node<N>,
}

fn build_hyperplane<R: Rng, const N: usize>(
    indexes: &[usize],
    all_vectors: &[Vector<N>],
    rng: &mut R,
) -> (HyperPlane<N>, Vec<usize>, Vec<usize>) {
    let mut sample_iter = indexes.choose_multiple(rng, 2);

    let p1 = &all_vectors[*sample_iter.next().unwrap()];
    let p2 = &all_vectors[*sample_iter.next().unwrap()];

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
        if hyperplane.point_is_above(&all_vectors[id]) {
            above.push(id);
        } else {
            below.push(id);
        }
    }

    (hyperplane, above, below)
}
