comet_rs::comet_fn! { dtranspose_mult_dense_dense, {
    let i = Index::with_value(4);
    let j = Index::with_value(4);
    let k = Index::with_value(4);

    let A = Tensor::<f64>::dense([i, j]).fill(3.2);
    let B = Tensor::<f64>::dense([j, i]).fill(0.0);
    let C = Tensor::<f64>::dense([i, k]).fill(1.0);
    let D = Tensor::<f64>::dense([j, k]).fill(0.0);

    B = A.transpose([j,i]);
    D = B * C;
    D.print();

},
}

fn main() {
    dtranspose_mult_dense_dense();
}
