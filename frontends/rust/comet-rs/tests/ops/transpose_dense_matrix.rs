comet_rs::comet_fn! { transpose_dense_mat, {
    let i = Index::with_value(4);
    let j = Index::with_value(4);

    let A = Tensor::<f64>::dense([i, j]).fill(3.2);
    let B = Tensor::<f64>::dense([j, i]).fill(0.0);
    B = A.transpose([j,i]);
    B.print();
},
}

fn main() {
    transpose_dense_mat();
}
