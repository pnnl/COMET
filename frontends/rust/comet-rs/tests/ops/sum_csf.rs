comet_rs::comet_fn! { sum_csf, {
    let i = Index::new();
    let j = Index::new();
    let k = Index::new();

    let A = Tensor::<f64>::csf([i, j, k]).load("../../../integration_test/data/test_rank3.tns");
    let a = A.sum();
    a.print();
}}

fn main() {
    sum_csf();
}


