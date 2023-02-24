comet_rs::comet_fn! { print_coo_multi, {
    let a = Index::new();
    let b = Index::new();
    let c = Index::new();
    let d = Index::new();

    let A = Tensor::<f64>::coo([a, b]).load("../../../integration_test/data/test_rank2_small.mtx");
    let B = Tensor::<f64>::coo([c, d]).load("../../../integration_test/data/test_rank2.mtx");

    A.print();
    B.print();
}}

fn main() {
    print_coo_multi();
}
