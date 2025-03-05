# VectorLite
The SQLite of Vector Database in Rust.

## Features
- [x] Compact size
- [x] Zero-copy de/serialization from/to disk 
- [x] Concurrent insert/delete/search
- [x] Excellent search performance
- [x] WASM support

## Benchmark

```bash
cargo bench --bench ann_benchmark "ANN Index Building/10000"
```
