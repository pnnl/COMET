comet_rs::comet_fn! { elwise_dense_csr_dense, {
    let a = Index::new();
    let b = Index::new();

    let B = Tensor::<f64>::csr([a, b]).load("../../../integration_test/data/test_rank2.mtx");
    let A = Tensor::<f64>::dense([a, b]).fill(2.7);
    let C = Tensor::<f64>::dense([a, b]).fill(0.0);

    C = A .* B;
    C.print();
}}

fn main() {
    elwise_dense_csr_dense();
}


