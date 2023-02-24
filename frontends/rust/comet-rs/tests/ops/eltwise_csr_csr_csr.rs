comet_rs::comet_fn! { elwise_csr_csr_csr, {
    let a = Index::new();
    let b = Index::new();

    let A = Tensor::<f64>::csr([a, b]).load("SPARSE_FILE_NAME");
    let B = Tensor::<f64>::csr([a, b]).load("SPARSE_FILE_NAME");
    let C = Tensor::<f64>::csr([a, b]);

    C = A .* B;
    C.print();
}}

fn main() {
    elwise_csr_csr_csr();
}

