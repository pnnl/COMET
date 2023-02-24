comet_rs::comet_fn! { transpose_coo_tensor, {
    let i = Index::new();
    let j = Index::new();
    let k = Index::new();

    let A = Tensor::<f64>::coo([i, j, k]).load("../../../integration_test/data/test_rank3.tns");
    let B = Tensor::<f64>::coo([j, i, k]);

    B = A.transpose([j,i,k]);
    B.print();
}}

fn main() {
    transpose_coo_tensor();
}