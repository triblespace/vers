use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;

mod common;

fn bench_nth_child(b: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut group = b.benchmark_group("DFUDS nth child Benchmark: Randomized Input");
    group.plot_config(common::plot_config());

    for l in common::LIMITED_SIZES {
        let (tree, nodes, number_children) = common::construct_random_tree(&mut rng, l);

        let sample = Uniform::new(0, number_children.len());
        group.bench_with_input(BenchmarkId::new("child", l), &l, |b, _| {
            b.iter_batched(
                || {
                    let mut i = sample.sample(&mut rng);
                    while number_children[i] == 0 {
                        i = sample.sample(&mut rng);
                    }

                    (nodes[i], rng.gen_range(0..number_children[i]))
                },
                |(i, n)| black_box(tree.nth_child(i, n)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_nth_child);
criterion_main!(benches);
