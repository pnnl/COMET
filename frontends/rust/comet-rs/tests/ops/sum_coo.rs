comet_rs::comet_fn! { sum_coo, {
    let i = Index::new();
    let j = Index::new();

    let A = Tensor::<f64>::coo([i, j]).fill_from_file("../../../integration_test/data/test_rank2.mtx");
    let a = A.sum();
    a.print();
}}

fn main() {
    sum_coo();
}


