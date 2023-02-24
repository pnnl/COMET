comet_rs::comet_fn! { transpose_csf_tensor, {
    let i = Index::new();
    let j = Index::new();
    let k = Index::new();

    let A = Tensor::<f64>::csf([i, j, k]).load("../../../integration_test/data/test_rank3.tns");
    let B = Tensor::<f64>::csf([k, i, j]);

    B = A.transpose([k,i,j]);
    B.print();
}}

fn main() {
    transpose_csf_tensor();
}


