comet_rs::comet_fn! { ccsd_t1_21_ttgt_tiling, {
    let i, c = Index::with_value(2);
    let m, n, a = Index::with_value(4);

    let v = Tensor::<f64>::dense([i, c, m, n]).fill(2.3);
    let t2 = Tensor::<f64>::dense([m, n, c, a]).fill(3.4);
    let i0 = Tensor::<f64>::dense([i, a]).fill(0.0);
    i0 = v * t2;
    i0.print();
},
CometOption::[MatMulTiling, TcToTtgt, ToLlvm]}
fn main() {
    ccsd_t1_21_ttgt_tiling();
}


