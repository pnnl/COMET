comet_rs::comet_fn! { sp_transpose_csr_eltwise_csr_csr, {
    let i = Index::new();
    let j = Index::new();

    let A = Tensor::<f64>::csr([i, j]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let B = Tensor::<f64>::csr([j, i]);
    let C = Tensor::<f64>::csr([j, i]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let D = Tensor::<f64>::csr([j, i]);

    B = A.transpose([j,i]);
    D = B .* C;
    D.print();

}}

fn main() {
    sp_transpose_csr_eltwise_csr_csr();
}
