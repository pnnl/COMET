comet_rs::comet_fn! { dense_ij_ikj_kj, {
    let i = Index::with_value(4);
    let k = Index::with_value(4);
    let j = Index::with_value(4);

    let A = Tensor::<f64>::dense([i,k,j]).fill(3.2);
    let B = Tensor::<f64>::dense([k,j]).fill(1.7);
    let mut C = Tensor::<f64>::dense([i,j]).fill(0.0);
    C = A * B;
    C.print();
}}

fn main() {
    dense_ij_ikj_kj();
}


