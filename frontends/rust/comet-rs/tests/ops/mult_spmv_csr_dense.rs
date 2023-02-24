comet_rs::comet_fn! { sparse_v_csr_dense, {
    let a = Index::new();
    let b = Index::new();

    let A = Tensor::<f64>::csr([a, b]).load("../../../integration_test/data/test_rank2.mtx");
    let B = Tensor::<f64>::dense([b]).fill(1.7);
    let C = Tensor::<f64>::dense([a]).fill(0.0);
    C = A * B;
    C.print();
}}

fn main() {
    sparse_v_csr_dense();
}


