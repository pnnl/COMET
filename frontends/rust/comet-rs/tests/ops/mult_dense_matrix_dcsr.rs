comet_rs::comet_fn! { dense_mat_dcsr, {
    let a = Index::with_value(4);
    let b = Index::new();
    let c = Index::new();

    let B = Tensor::<f64>::dcsr([b,c]).load("../../../integration_test/data/test_rank2.mtx");
    let A = Tensor::<f64>::dense([a,b]).fill(1.7);
    let mut C = Tensor::<f64>::dense([a,c]).fill(0.0);
    C = A * B;
    C.print();
}}

fn main() {
    dense_mat_dcsr();
}


