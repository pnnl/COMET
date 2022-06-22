comet_rs::comet_fn! { elwise_monoid_min_dense_dense_dense, {
    let a = Index::with_value(4);
    let b = Index::with_value(4);

    let A = Tensor::<f64>::dense([a, b]).fill(2.7);
    let B = Tensor::<f64>::dense([a, b]).fill(3.2);
    let C = Tensor::<f64>::dense([a, b]).fill(0.0);
    C = A @(min) B;
    C.print();
}}

fn main() {
    elwise_monoid_min_dense_dense_dense();
}


