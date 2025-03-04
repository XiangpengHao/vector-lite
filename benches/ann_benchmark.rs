use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vector_search::{ANNIndex, Vector};

fn gen_vector<const N: usize>(rng: &mut StdRng) -> Vector<N> {
    let coords: [f32; N] = std::array::from_fn(|_| rng.random_range(-1.0..1.0));
    Vector::new(coords)
}

// Utility function to generate random vectors
fn generate_random_vectors<const N: usize>(count: usize, seed: u64) -> (Vec<Vector<N>>, Vec<i32>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let vectors: Vec<Vector<N>> = (0..count).map(|_| gen_vector(&mut rng)).collect();

    let ids: Vec<i32> = (0..count as i32).collect();

    (vectors, ids)
}

const MAX_LEAF_SIZE: i32 = 15;
const NUM_TREES: i32 = 3;

// Benchmark for building index
fn bench_build_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("ANN Index Building");

    const DIM: usize = 768;

    let sizes = [1_000, 10_000, 100_000];

    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_with_setup(
                || {
                    let (vectors, ids) = generate_random_vectors::<DIM>(size, 42);
                    (vectors, ids)
                },
                |(vectors, ids)| {
                    let mut seed_rng = StdRng::seed_from_u64(42);
                    black_box(ANNIndex::build_index(
                        NUM_TREES,
                        MAX_LEAF_SIZE,
                        &vectors,
                        &ids,
                        &mut seed_rng,
                    ));
                },
            );
        });
    }

    group.finish();
}

// Benchmark for searching
fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("ANN Search");

    const DIM: usize = 768;
    let sizes = [1_000, 10_000, 100_000];

    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            // Setup: Build the index once
            let (vectors, ids) = generate_random_vectors::<DIM>(size, 42);
            let mut seed_rng = StdRng::seed_from_u64(42);
            let index =
                ANNIndex::build_index(NUM_TREES, MAX_LEAF_SIZE, &vectors, &ids, &mut seed_rng);

            // Generate a random query vector
            let mut query_rng = StdRng::seed_from_u64(123);
            let query_coords: [f32; DIM] =
                std::array::from_fn(|_| query_rng.random_range(f32::MIN..=f32::MAX));
            let query = Vector::new(query_coords);

            // Benchmark the search
            b.iter(|| {
                black_box(index.search(&query, 20));
            });
        });
    }

    group.finish();
}

// Function for running the full-scale benchmark (1M vectors)
// Warning: This might take a long time to run
fn bench_full_scale(c: &mut Criterion) {
    // Only run this benchmark when explicitly requested
    if std::env::var("RUN_FULL_SCALE").is_ok() {
        let mut group = c.benchmark_group("Full Scale ANN (1M vectors)");
        group.sample_size(10); // Use fewer samples due to the long runtime

        const DIM: usize = 768;
        const SIZE: usize = 100_000;

        // Benchmark index building
        group.bench_function("Build Index", |b| {
            b.iter_with_setup(
                || {
                    let (vectors, ids) = generate_random_vectors::<DIM>(SIZE, 42);
                    (vectors, ids)
                },
                |(vectors, ids)| {
                    let mut seed_rng = StdRng::seed_from_u64(42);
                    black_box(ANNIndex::build_index(
                        NUM_TREES,
                        MAX_LEAF_SIZE,
                        &vectors,
                        &ids,
                        &mut seed_rng,
                    ));
                },
            );
        });

        // Benchmark search (after building the index once)
        group.bench_function("Search Top 20", |b| {
            // Setup: Build the index once
            let (vectors, ids) = generate_random_vectors::<DIM>(SIZE, 42);
            let mut seed_rng = StdRng::seed_from_u64(42);
            let index =
                ANNIndex::build_index(NUM_TREES, MAX_LEAF_SIZE, &vectors, &ids, &mut seed_rng);

            // Generate a random query vector
            let mut query_rng = StdRng::seed_from_u64(123);
            let query_coords: [f32; DIM] =
                std::array::from_fn(|_| query_rng.random_range(f32::MIN..=f32::MAX));
            let query = Vector::new(query_coords);

            // Benchmark the search
            b.iter(|| {
                black_box(index.search(&query, 20));
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_build_index, bench_search, bench_full_scale);
criterion_main!(benches);
