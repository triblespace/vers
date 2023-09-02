use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Uniform};

mod common;

fn bench_parent(b: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut group = b.benchmark_group("DFUDS Parent Benchmark: Randomized Input");
    group.plot_config(common::plot_config());

    for l in common::LIMITED_SIZES {
        let (tree, nodes, _) = common::construct_random_tree(&mut rng, l);

        let sample = Uniform::new(1, nodes.len());
        group.bench_with_input(BenchmarkId::new("parent", l), &l, |b, _| {
            b.iter_batched(
                || nodes[sample.sample(&mut rng)],
                |e| black_box(tree.parent(e)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_parent);
criterion_main!(benches);
