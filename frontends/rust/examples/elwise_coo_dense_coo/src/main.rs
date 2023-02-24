comet_rs::comet_fn! { elwise_coo_dense_coo, {
    let a = Index::new();
    let b = Index::new();

    let A = Tensor::<f64>::coo([a, b]).load("test_rank2.mtx");
    let B = Tensor::<f64>::dense([a, b]).fill(2.7);
    let C = Tensor::<f64>::coo([a, b]);

    C = A .* B;
    C.print();
}}

fn main() {
    elwise_coo_dense_coo();
}

//need to set
// COMET_BIN_DIR= (where comet binaries were installed)
// COMET_LIB_DIR=(where comet libraries were installed)
// MLIR_BIN_DIR= (where mlir binaries were installed)
// MLIR_LIB_DIR= (where mlir binaries were installed)

/*
expected output
"
data =
0,9,
data =
0,0,1,1,2,3,3,4,4,
data =
0,
data =
0,3,1,4,2,0,3,1,4,
data =
2.7,3.78,5.4,6.75,8.1,11.07,10.8,14.04,13.5,
",
*/
