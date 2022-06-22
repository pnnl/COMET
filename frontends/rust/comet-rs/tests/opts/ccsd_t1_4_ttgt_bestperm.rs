comet_rs::comet_fn! { ccsd_t1_4_ttgt_bestperm, {
    let i, c = Index::with_value(2);
    let m, a = Index::with_value(4);

    let v = Tensor::<f64>::dense([c, i, m, a]).fill(2.3);
    let t2 = Tensor::<f64>::dense([m, c]).fill(3.4);
    let i0 = Tensor::<f64>::dense([i, a]).fill(0.0);
    i0 = v * t2;
    i0.print();
},
CometOption::[BestPermTtgt, TcToTtgt],
MlirOption::[ConvertLinalgToLoops, ConvertLinalgToStd, ConvertLinalgToLlvm, ConvertScfToStd, ConvertStdToLlvm]
}

fn main() {
    ccsd_t1_4_ttgt_bestperm();
}


