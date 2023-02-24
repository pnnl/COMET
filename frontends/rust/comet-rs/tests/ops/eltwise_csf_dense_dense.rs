comet_rs::comet_fn! { elwise_csf_dense_dense, {
    let a = Index::new();
    let b = Index::new();
    let c = Index::new();

    let A = Tensor::<f64>::csf([a, b, c]).load("../../../integration_test/data/test_rank3.tns");
    let B = Tensor::<f64>::dense([a, b, c]).fill(2.7);
    let C = Tensor::<f64>::dense([a, b, c]).fill(0.0);

    C = A .* B;
    C.print();
}}

fn main() {
    elwise_csf_dense_dense();
}


