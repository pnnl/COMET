comet_rs::comet_fn! { mm_semiring_plustimes_coo_dense_dense, {
    let a = Index::new();
    let b = Index::new();
    let c = Index::with_value(4);

    let A = Tensor::<f64>::coo([a, b]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let B = Tensor::<f64>::dense([b, c]).fill(1.7);
    let C = Tensor::<f64>::dense([a, c]).fill(0.0);
    C = A @(+,*) B;
    C.print();
}}

fn main() {
    mm_semiring_plustimes_coo_dense_dense();
}


