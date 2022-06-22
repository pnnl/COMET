comet_rs::comet_fn! { get_time, {
    let i = Index::with_value(128);
    let j = Index::with_value(128);
    let k = Index::with_value(128);

    let A = Tensor::<f64>::dense([i, j]).fill(3.5);
    let B = Tensor::<f64>::dense([j, k]).fill(2.1);
    let mut C = Tensor::<f64>::dense([i, k]).fill(0.0);
    let start = getTime();
    C = A * B;
    let end = getTime();
    printElapsedTime(start, end);
}}

fn main() {
    get_time();
}


