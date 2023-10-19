use comet_rs::*;

comet_fn! { ccsd_t1_3_ttgt, {
    let i, c = Index::with_value(2);
    let a = Index::with_value(4);

    let v = Tensor::<f64>::dense([c, a]).fill(2.3);
    let t2 = Tensor::<f64>::dense([i, c]).fill(3.4);
    let i0 = Tensor::<f64>::dense([i, a]).fill(0.0);
    i0 = v * t2;
    i0.print();
},
CometOption::[TcToTtgt, ToLoops, ToLlvm]}

fn main() {
    ccsd_t1_3_ttgt();
}


