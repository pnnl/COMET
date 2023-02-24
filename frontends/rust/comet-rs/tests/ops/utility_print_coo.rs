comet_rs::comet_fn! { print_coo, {
    let i = Index::new();
    let j = Index::new();

    let A = Tensor::<f64>::coo([i, j]).load("../../../integration_test/data/test_rank2_small.mtx");

    A.print();
}}

fn main() {
    print_coo();
}
