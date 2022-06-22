comet_rs::comet_fn! { elwise_dense_csf_dense, {
    let a = Index::new();
    let b = Index::new();
    let c = Index::new();

    let B = Tensor::<f64>::csf([a, b, c]).fill_from_file("../../../integration_test/data/test_rank3.tns");
    let A = Tensor::<f64>::dense([a, b, c]).fill(2.7);
    let C = Tensor::<f64>::dense([a, b, c]).fill(0.0);

    C = A .* B;
    C.print();
}}

fn main() {
    elwise_dense_csf_dense();
}
