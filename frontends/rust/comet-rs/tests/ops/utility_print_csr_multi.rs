comet_rs::comet_fn! { print_csr_multi, {
    let a = Index::new();
    let b = Index::new();
    let c = Index::new();
    let d = Index::new();
    let e = Index::new();
    let f = Index::new();

    let A = Tensor::<f64>::csr([a, b]).fill_from_file("../../../integration_test/data/test_rank2_small.mtx");
    let B = Tensor::<f64>::csr([c, d]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let C = Tensor::<f64>::csr([e, f]).fill_from_file("../../../integration_test/data/test_rank2_small.mtx");

    A.print();
    B.print();
    C.print();
}}

fn main() {
    print_csr_multi();
}


