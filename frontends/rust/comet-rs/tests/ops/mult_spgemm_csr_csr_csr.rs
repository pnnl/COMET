comet_rs::comet_fn! { mult_spgemm_csr_csr_csr, {
    let a = Index::new();
    let b = Index::new();
    let c = Index::new();

    let A = Tensor::<f64>::csr([a, b]).load("/home/frie869/projects/COMET/integration_test/data/test_rank2.mtx");
    let B = Tensor::<f64>::csr([b, c]).load("/home/frie869/projects/COMET/integration_test/data/test_rank2copy.mtx"); 
    let C = Tensor::<f64>::csr([a, c]);
    C = A * B;
    C.print();
}}

fn main() {
    mult_spgemm_csr_csr_csr(); 
}


