comet_rs::comet_fn! { print_dense, {
    let i = Index::with_value(4);
    let j = Index::with_value(4);

    let A = Tensor::<f64>::dense([i, j]).fill(2.3);

    A.print();
}}

comet_rs::comet_fn! { sum_coo, {
    let i = Index::new();
    let j = Index::new();

    let A = Tensor::<f64>::coo([i, j]).load("../../../integration_test/data/test_rank2.mtx");
    let a = A.sum();
    a.print();
}}

comet_rs::comet_fn! { dense_mm, {
    let i = Index::with_value(8);
    let j: Index = Index::with_value(4);
    let k = Index::with_value(2);
    let A = Tensor::<f64>::dense([i, j]).fill(2.2);
    let mut B = Tensor::<f64>::dense([j, k]).fill(3.4);
    let mut C = Tensor::<f64>::dense([i, k]).fill(0.0);
    C = A * B;
    C.print();
}}

fn main() {
    print_dense();
    sum_coo();
    dense_mm();
}
