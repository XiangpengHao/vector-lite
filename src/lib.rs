#![doc = include_str!("../README.md")]
mod node;
use std::hash::Hash;

use node::Node;
mod external_index;
pub use external_index::{ANNIndexExternal, LinearSearchExternal, LshExternal};
mod vector;
pub use vector::Vector;
mod owned_index;
pub use owned_index::{ANNIndexOwned, VectorLite};

// TODO: no need for Hash
pub trait VectorKey:
    bincode::Encode + bincode::Decode + Clone + PartialEq + Eq + Hash + 'static
{
}

impl VectorKey for String {}

impl VectorKey for u32 {}
