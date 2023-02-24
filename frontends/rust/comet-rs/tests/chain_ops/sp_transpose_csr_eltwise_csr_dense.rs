comet_rs::comet_fn! { sp_transpose_csr_eltwise_csr_dense, {
    let i = Index::new();
    let j = Index::new();

    let A = Tensor::<f64>::csr([i, j]).load("../../../integration_test/data/test_rank2_small.mtx");
    let B = Tensor::<f64>::csr([j, i]);
    let C = Tensor::<f64>::dense([j, i]).fill(2.3);
    let D = Tensor::<f64>::dense([j, i]).fill(0.0);

    B = A.transpose([j,i]);
    D = B .* C;
    D.print();

}}

fn main() {
    sp_transpose_csr_eltwise_csr_dense();
}


