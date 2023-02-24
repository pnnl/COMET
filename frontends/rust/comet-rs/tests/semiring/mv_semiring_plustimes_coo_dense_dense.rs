comet_rs::comet_fn! { mv_semiring_plustimes_coo_dense_dense, {
    let a = Index::new();
    let b = Index::new();

    let A = Tensor::<f64>::coo([a, b]).load("../../../integration_test/data/test_rank2.mtx");
    let B = Tensor::<f64>::dense([b]).fill(1.7);
    let C = Tensor::<f64>::dense([a]).fill(0.0);
    C = A @(+,*) B;
    C.print();
}}

fn main() {
    mv_semiring_plustimes_coo_dense_dense();
}


