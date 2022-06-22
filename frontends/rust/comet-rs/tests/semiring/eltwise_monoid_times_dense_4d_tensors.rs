comet_rs::comet_fn! { elwise_monoid_times_dense_4d_tensors, {
    let a = Index::with_value(2);
    let b = Index::with_value(2);
    let c = Index::with_value(2);
    let d = Index::with_value(2);

    let A = Tensor::<f64>::dense([a, b, c, d]).fill(2.2);
    let B = Tensor::<f64>::dense([a, b, c, d]).fill(3.6);
    let C = Tensor::<f64>::dense([a, b, c, d]).fill(0.0);
    C = A @(*) B;
    C.print();
}}

fn main() {
    elwise_monoid_times_dense_4d_tensors();
}


