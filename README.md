# VectorLite
[![Rust CI](https://github.com/XiangpengHao/vector-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/XiangpengHao/vector-lite/actions/workflows/ci.yml)
[![Crates.io Version](https://img.shields.io/crates/v/vector-lite)](https://crates.io/crates/vector-lite)
[![docs.rs](https://img.shields.io/docsrs/vector-lite)](https://docs.rs/vector-lite/latest/vector_lite/)

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
let async_task = async {
index.insert([1.0, 0.0, 0.0].into(), "101".to_string()).await;
index.insert([0.0, 1.0, 0.0].into(), "102".to_string()).await;
index.insert([0.0, 0.0, 1.0].into(), "103".to_string()).await;
index.insert([0.7, 0.7, 0.0].into(), "104".to_string()).await;

let query = [0.9, 0.1, 0.0].into();
let results = index.search(&query, 2).await;

for (id, distance) in results {
    println!("ID: {}, Distance: {}", id, distance);
}

index.delete_by_id(&"102").await;

// De/serialize from/to disk
let serialized = index.to_bytes();
let loaded_index = VectorLite::<DIM>::from_bytes(&serialized);
};

let rt = tokio::runtime::Runtime::new().unwrap();
rt.block_on(async_task);
```

## Benchmark

```bash
cargo bench --bench ann_benchmark "ANN Index Building/10000"
```

## WASM

Need to build with [`wasm_js` backend](https://docs.rs/getrandom/latest/getrandom/#webassembly-support) for `getrandom` crate.
```bash
env RUSTFLAGS='--cfg getrandom_backend="wasm_js"' cargo build --target wasm32-unknown-unknown
```

Run tests:
```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Run test
env RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack test --headless --firefox
```

## Notable users
VectorLite is the primary vector database used in [Seen](https://github.com/XiangpengHao/seen) -- knowledge management for the impatient. 

## License
Apache-2.0
