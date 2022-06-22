comet_rs::comet_fn! { print_csr, {
    let i = Index::new();
    let j = Index::new();

    let A = Tensor::<f64>::csr([i, j]).fill_from_file("../../../integration_test/data/test_rank2_small.mtx");

    A.print();
}}

fn main() {
    print_csr();
}


