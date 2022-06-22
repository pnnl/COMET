comet_rs::comet_fn! { elwise_dcsr_dense_dense, {
    let a = Index::new();
    let b = Index::new();

    let A = Tensor::<f64>::dcsr([a, b]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let B = Tensor::<f64>::dense([a, b]).fill(2.7);
    let C = Tensor::<f64>::dense([a, b]).fill(0.0);

    C = A .* B;
    C.print();
}}

fn main() {
    elwise_dcsr_dense_dense();
}


