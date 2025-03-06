use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vector_lite::{
    ANNIndexExternal, ANNIndexOwned, LinearSearchExternal, LshExternal, Vector, VectorLite,
};

fn gen_vector<const N: usize>(rng: &mut StdRng) -> Vector<N> {
    let coords: [f32; N] = std::array::from_fn(|_| rng.random_range(-1.0..1.0));
    Vector::from(coords)
}

// Utility function to generate random vectors
fn generate_random_vectors<const N: usize>(count: usize, seed: u64) -> Vec<Vector<N>> {
    let mut rng = StdRng::seed_from_u64(seed);

    let vectors: Vec<Vector<N>> = (0..count).map(|_| gen_vector(&mut rng)).collect();

    vectors
}

const DIM: usize = 768;
const MAX_LEAF_SIZE: usize = 30;
const NUM_TREES: usize = 3;

// Benchmark for building index
fn bench_build_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("ANN Index Building");

    let sizes = [1_000, 10_000, 100_000];

    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_with_setup(
                || generate_random_vectors::<DIM>(size, 42),
                |vectors| {
                    let mut seed_rng = StdRng::seed_from_u64(42);
                    black_box(
                        LshExternal::build(NUM_TREES, MAX_LEAF_SIZE, &vectors, &mut seed_rng)
                            .unwrap(),
                    );
                },
            );
        });
    }

    group.finish();
}

// Benchmark for searching
fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("ANN Search");

    let sizes = [1_000, 10_000, 100_000];

    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            // Setup: Build the index once
            let vectors = generate_random_vectors::<DIM>(size, 42);
            let mut seed_rng = StdRng::seed_from_u64(42);
            let index =
                LshExternal::build(NUM_TREES, MAX_LEAF_SIZE, &vectors, &mut seed_rng).unwrap();

            // Generate a random query vector
            let mut query_rng = StdRng::seed_from_u64(123);
            let query = gen_vector::<DIM>(&mut query_rng);

            // Benchmark the search
            b.iter(|| {
                black_box(index.search(&query, 20));
            });
        });
    }

    group.finish();
}

// Benchmark for searching
fn bench_linear_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("ANN Linear Search");

    let sizes = [1_000, 10_000, 100_000];

    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            // Setup: Build the index once
            let vectors = generate_random_vectors::<DIM>(size, 42);
            let mut seed_rng = StdRng::seed_from_u64(42);
            let index =
                LinearSearchExternal::build(NUM_TREES, MAX_LEAF_SIZE, &vectors, &mut seed_rng)
                    .unwrap();

            // Generate a random query vector
            let mut query_rng = StdRng::seed_from_u64(123);
            let query = gen_vector::<DIM>(&mut query_rng);

            // Benchmark the search
            b.iter(|| {
                black_box(index.search(&query, 20));
            });
        });
    }

    group.finish();
}

fn bench_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("VectorLite serialize");

    let sizes = [1_000, 10_000, 100_000];

    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vectors = generate_random_vectors::<DIM>(size, 42);
            let mut seed_rng = StdRng::seed_from_u64(42);

            let mut index = VectorLite::<DIM>::new(NUM_TREES, MAX_LEAF_SIZE);
            for (i, vector) in vectors.iter().enumerate() {
                index.insert_with_rng(vector.clone(), format!("{}", i), &mut seed_rng);
            }

            b.iter(|| {
                black_box(index.to_bytes());
            });
        });
    }

    group.finish();
}

fn bench_deserialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("VectorLite deserialize");

    let sizes = [1_000, 10_000, 100_000];

    for size in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let vectors = generate_random_vectors::<DIM>(size, 42);
            let mut seed_rng = StdRng::seed_from_u64(42);

            let mut index = VectorLite::<DIM>::new(NUM_TREES, MAX_LEAF_SIZE);
            for (i, vector) in vectors.iter().enumerate() {
                index.insert_with_rng(vector.clone(), format!("{}", i), &mut seed_rng);
            }

            let bytes = index.to_bytes();

            b.iter(|| {
                black_box(VectorLite::<DIM>::from_bytes(&bytes));
            });
        });
    }
}

criterion_group!(
    benches,
    bench_build_index,
    bench_search,
    bench_linear_search,
    bench_serialize,
    bench_deserialize
);
criterion_main!(benches);
