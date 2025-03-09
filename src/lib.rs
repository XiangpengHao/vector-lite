#![doc = include_str!("../README.md")]
mod node;
use std::{hash::Hash, rc::Rc};

use node::Node;
mod external_index;
pub use external_index::{ANNIndexExternal, LinearSearchExternal, LshExternal};
mod vector;
pub use vector::Vector;
mod owned_index;
pub use owned_index::{ANNIndexOwned, ScoreMetric, VectorLite, VectorLiteIndex};

pub trait VectorKey: bincode::Encode + Clone + PartialEq + Eq + Hash + 'static {}

impl VectorKey for String {}

impl VectorKey for u32 {}

impl VectorKey for Rc<String> {}

