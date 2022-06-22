comet_rs::comet_fn! { print_dense, {
    let i = Index::with_value(4);
    let j = Index::with_value(4);

    let A = Tensor::<f64>::dense([i, j]).fill(2.3);

    A.print();
}}

fn main() {
    print_dense();
}


