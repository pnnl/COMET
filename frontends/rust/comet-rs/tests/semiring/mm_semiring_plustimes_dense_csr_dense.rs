comet_rs::comet_fn! { mm_semiring_plustimes_dense_csr_dense, {
    let a = Index::with_value(4);
    let b = Index::new();
    let c = Index::new();

    let B = Tensor::<f64>::csr([b, c]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let A = Tensor::<f64>::dense([a, b]).fill(1.7);
    let C = Tensor::<f64>::dense([a, c]).fill(0.0);
    C = A @(+,*) B;
    C.print();
}}

fn main() {
    mm_semiring_plustimes_dense_csr_dense();
}


