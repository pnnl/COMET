comet_rs::comet_fn! { mm_semiring_plustimes_dense_4d_tensors, {
    let a = Index::with_value(2);
    let b = Index::with_value(2);
    let c = Index::with_value(2);
    let d = Index::with_value(2);
    let e = Index::with_value(2);
    let f = Index::with_value(2);

    let A = Tensor::<f64>::dense([a, e, d, f]).fill(2.2);
    let B = Tensor::<f64>::dense([b, f, c, e]).fill(3.6);
    let C = Tensor::<f64>::dense([a, b, c, d]).fill(0.0);
    C = A @(+,*) B;
    C.print();
}}

fn main() {
    mm_semiring_plustimes_dense_4d_tensors();
}


