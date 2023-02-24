comet_rs::comet_fn! { print_csf_multi, {
    let a = Index::new();
    let b = Index::new();
    let c = Index::new();
    let d = Index::new();
    let e = Index::new();
    let f = Index::new();

    let A = Tensor::<f64>::csf([a, b, c]).load("../../../integration_test/data/test_rank3.tns");
    let B = Tensor::<f64>::csf([d, e, f]).load("../../../integration_test/data/test_rank3.tns");

    A.print();
    B.print();
}}

fn main() {
    print_csf_multi();
}


