comet_rs::comet_fn! { dense_vec_dcsr, {
    let a = Index::new();
    let b = Index::new();

    let B = Tensor::<f64>::coo([a,b]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let A = Tensor::<f64>::dense([a]).fill(1.7);
    let mut C = Tensor::<f64>::dense([b]).fill(0.0);
    C = A * B;
    C.print();
}}

fn main() {
    dense_vec_dcsr();
}


