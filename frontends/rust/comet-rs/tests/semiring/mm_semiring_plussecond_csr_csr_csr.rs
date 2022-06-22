comet_rs::comet_fn! { mm_semiring_plussecond_csr_csr_csr, {
    let a = Index::new();
    let b = Index::new();
    let c = Index::new();

    let A = Tensor::<f64>::csr([a, b]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let B = Tensor::<f64>::csr([b, c]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let C = Tensor::<f64>::csr([a, c]);
    C = A @(+,second) B;
    C.print();
}}

fn main() {
    mm_semiring_plussecond_csr_csr_csr();
}


