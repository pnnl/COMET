comet_rs::comet_fn! { mv_semiring_plustimes_dense_dcsr_dense, {
    let a = Index::new();
    let b = Index::new();

    let B = Tensor::<f64>::dcsr([a,b]).load("../../../integration_test/data/test_rank2.mtx");
    let A = Tensor::<f64>::dense([a]).fill(1.7);
    let C = Tensor::<f64>::dense([b]).fill(0.0);
    C = A @(+,*) B;
    C.print();
}}

fn main() {
    mv_semiring_plustimes_dense_dcsr_dense();
}


