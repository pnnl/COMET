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
    dense_mm();
}

