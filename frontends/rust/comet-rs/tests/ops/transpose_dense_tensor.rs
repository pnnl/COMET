comet_rs::comet_fn! { transpose_dense_tensor, {
    let i = Index::with_value(4);
    let j = Index::with_value(4);
    let k = Index::with_value(4);

    let A = Tensor::<f64>::dense([i, j, k]).fill(3.7);
    let B = Tensor::<f64>::dense([j, i, k]).fill(0.0);
    B = A.transpose([j, i, k]);
    B.print();
}
}

fn main() {
    transpose_dense_tensor();
}
