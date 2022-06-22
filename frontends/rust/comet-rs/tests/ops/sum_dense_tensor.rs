comet_rs::comet_fn! { sum_dense_tensor, {
    let i = Index::with_value(4);
    let j = Index::with_value(4);
    let k = Index::with_value(4);

    let A = Tensor::<f64>::dense([i, j, k]).fill(3.7);
    let a = A.sum();
    a.print();
}}

fn main() {
    sum_dense_tensor();
}


