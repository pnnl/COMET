comet_rs::comet_fn! { sp_transpose_csr_spmm_csr_dense, {
    let i = Index::new();
    let j = Index::new();
    let k = Index::with_value(5);

    let A = Tensor::<f64>::csr([i, j]).load("../../../integration_test/data/test_rank2.mtx");
    let B = Tensor::<f64>::csr([j, i]);
    let C = Tensor::<f64>::dense([i, k]).fill(1.0);
    let D = Tensor::<f64>::dense([j, k]).fill(0.0);

    B = A.transpose([j,i]);
    D = B * C;
    D.print();

}}

fn main() {
    sp_transpose_csr_spmm_csr_dense();
}