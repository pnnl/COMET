comet_rs::comet_fn! { transpose_coo_mat, {
    let i = Index::new();
    let j = Index::new();

    let A = Tensor::<f64>::coo([i, j]).load("../../../integration_test/data/test_rank2.mtx");
    let B = Tensor::<f64>::coo([j, i]);

    B = A.transpose([j,i]);
    B.print();
}}

fn main() {
    transpose_coo_mat();
}


