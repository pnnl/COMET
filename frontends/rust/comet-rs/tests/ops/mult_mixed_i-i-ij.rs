comet_rs::comet_fn! { mixed_i_i_ij, {
    let i = Index::new();
    let j = Index::new();

    let B = Tensor::<f64>::coo([i,j]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let A = Tensor::<f64>::dense([i]).fill(1.7);
    let mut C = Tensor::<f64>::dense([i]).fill(0.0);
    C = A * B;
    C.print();
}}

fn main() {
    mixed_i_i_ij();
}
