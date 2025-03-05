# VectorLite
The SQLite of Vector Database in Rust.

## Features
- [x] Compact size
- [x] Zero-copy de/serialization from/to disk 
- [x] Excellent search performance
- [x] WASM support
- [x] Minimal dependencies

### Planned
- [ ] Concurrent insert/delete/search
- [ ] Support usage as external index for Parquet dataset.

## Usage 
```rust
use vector_lite::{VectorLite, Vector, ANNIndexOwned};

// Create a new index with 4 trees and leaf size of 10
const DIM: usize = 3;
let mut index = VectorLite::<DIM>::new(4, 10);
index.insert([1.0, 0.0, 0.0].into(), 101);
index.insert([0.0, 1.0, 0.0].into(), 102);
index.insert([0.0, 0.0, 1.0].into(), 103);
index.insert([0.7, 0.7, 0.0].into(), 104);

let query = [0.9, 0.1, 0.0].into();
let results = index.search(&query, 2);

for (id, distance) in results {
    println!("ID: {}, Distance: {}", id, distance);
}

index.delete_by_id(102);

// De/serialize from/to disk
let serialized = index.to_bytes();
let loaded_index = VectorLite::<DIM>::from_bytes(&serialized);
```

## Benchmark

```bash
cargo bench --bench ann_benchmark "ANN Index Building/10000"
```
