comet_rs::comet_fn! { elwise_monoid_times_coo_dense_coo, {
    let a = Index::new();
    let b = Index::new();

    let A = Tensor::<f64>::coo([a, b]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let B = Tensor::<f64>::dense([a, b]).fill(2.7);
    let C = Tensor::<f64>::coo([a, b]);
    C = A @(*) B;
    C.print();
}}

fn main() {
    elwise_monoid_times_coo_dense_coo();
}


