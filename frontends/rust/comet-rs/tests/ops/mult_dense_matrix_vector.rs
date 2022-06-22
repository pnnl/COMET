comet_rs::comet_fn! { dense_mv, {
    let i = Index::with_value(8);
    let j = Index::with_value(16);

    let A = Tensor::<f64>::dense([i, j]).fill(2.3);
    let B = Tensor::<f64>::dense([j]).fill(3.7);
    let mut C = Tensor::<f64>::dense([i]).fill(0.0);
    C = A * B;
    C.print();
}}

fn main() {
    dense_mv();
}


