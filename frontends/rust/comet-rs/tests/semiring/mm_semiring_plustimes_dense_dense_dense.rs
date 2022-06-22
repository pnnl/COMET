comet_rs::comet_fn! { mm_semiring_plustimes_dense_dense_dense, {
    let i = Index::with_value(8);
    let j = Index::with_value(4);
    let k = Index::with_value(2);

    let A = Tensor::<f64>::dense([i, j]).fill(2.2);
    let B = Tensor::<f64>::dense([j, k]).fill(3.4);
    let C = Tensor::<f64>::dense([i, k]).fill(0.0);
    C = A @(+,*) B;
    C.print();
}}

fn main() {
    mm_semiring_plustimes_dense_dense_dense();
}
