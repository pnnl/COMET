comet_rs::comet_fn! { opt_dense_transpose, {
    let i = Index::with_value(2);
    let j = Index::with_value(4);
    let k = Index::with_value(8);
    let l = Index::with_value(16);

    let A = Tensor::<f64>::dense([i, j, k, l]).fill(3.7);
    let B = Tensor::<f64>::dense([i, k, j, l]).fill(0.0);
    B = A.transpose([i, k, j, l]);
    B.print();
},
CometOption::[ DenseTranspose, TaToIt, ToLoops, ToLlvm]
}

fn main() {
    opt_dense_transpose();
}
