use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Uniform};

mod common;

fn bench_subtree(b: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut group = b.benchmark_group("DFUDS Subtree Size Benchmark: Randomized Input");
    group.plot_config(common::plot_config());

    for l in common::SIZES {
        let (tree, nodes, _) = common::construct_random_tree(&mut rng, l, 100);

        let sample = Uniform::new(1, nodes.len());
        group.bench_with_input(BenchmarkId::new("subtree_size", l), &l, |b, _| {
            b.iter_batched(
                || nodes[sample.sample(&mut rng)],
                |e| black_box(tree.subtree_size(e)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_subtree);
criterion_main!(benches);
