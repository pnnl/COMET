comet_rs::comet_fn! { mm_semiring_plustimes_csr_csr_csr, {
    let a = Index::new();
    let b = Index::new();
    let c = Index::new();

    let A = Tensor::<f64>::csr([a, b]).load("../../../integration_test/data/test_rank2.mtx");
    let B = Tensor::<f64>::csr([b, c]).load("../../../integration_test/data/test_rank2.mtx");
    let C = Tensor::<f64>::csr([a, c]);
    C = A @(+,*) B;
    C.print();
}}

fn main() {
    mm_semiring_plustimes_csr_csr_csr();
}


