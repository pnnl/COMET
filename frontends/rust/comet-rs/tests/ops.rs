use assert_cmd::Command;
use std::env;

const EX_DIR: &str = "./target/release/examples/";

fn remove_whitespace(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}
fn compare_strings(a: &str, b: &str) {
    let a = remove_whitespace(a);
    let b = remove_whitespace(b);
    assert_eq!(a, b);
}

#[test]
fn eltwise_coo_dense_coo() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_coo_dense_coo").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
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
    );
}

// #[test]
// fn eltwise_csf_dense_csf() {
//     let output = Command::new(EX_DIR.to_owned() + "eltwise_csf_dense_csf").unwrap();
//     compare_strings(
//         &String::from_utf8_lossy(&output.stdout),
//         "
//         data =
//         0,0,
//         data =
//         0,0,0,
//         data =
//         0,0,0,0,
//         data =
//         0,0,0,
//         data =
//         0,0,0,0,
//         data =
//         0,0,0,
//         data =
//         3.51,5.697,8.1,
//         ",
//     );
// }

// #[test]
// fn eltwise_csf_dense_dense() {
//     let output = Command::new(EX_DIR.to_owned() + "eltwise_csf_dense_dense").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"
//     data =
//     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5.697,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     ");
// }

#[test]
fn eltwise_csr_dense_csr() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_csr_dense_csr").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        5,
        data =
        0,
        data =
        0,2,4,5,7,9,
        data =
        0,3,1,4,2,0,3,1,4,
        data =
        2.7,3.78,5.4,6.75,8.1,11.07,10.8,14.04,13.5,
        ",
    );
}

#[test]
fn eltwise_csr_dense_dense() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_csr_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        2.7,0,0,3.78,0,0,5.4,0,0,6.75,0,0,8.1,0,0,11.07,0,0,10.8,0,0,14.04,0,0,13.5,
        ",
    );
}

#[test]
fn eltwise_dcsr_dense_dcsr() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_dcsr_dense_dcsr").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        0,5,
        data =
        0,1,2,3,4,
        data =
        0,2,4,5,7,9,
        data =
        0,3,1,4,2,0,3,1,4,
        data =
        2.7,3.78,5.4,6.75,8.1,11.07,10.8,14.04,13.5,
        ",
    );
}

#[test]
fn eltwise_dcsr_dense_dense() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_dcsr_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        2.7,0,0,3.78,0,0,5.4,0,0,6.75,0,0,8.1,0,0,11.07,0,0,10.8,0,0,14.04,0,0,13.5,
        ",
    );
}

#[test]

fn eltwise_dense_4d_tensors() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_dense_4d_tensors").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,
        ",
    );
}

// #[test]
// fn eltwise_dense_csf_dense() {
//     let output = Command::new(EX_DIR.to_owned() + "eltwise_dense_csf_dense").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"
//     data =
//     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5.697,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     ");
// }

#[test]
fn eltwise_dense_csr_dense() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_dense_csr_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        2.7,0,0,3.78,0,0,5.4,0,0,6.75,0,0,8.1,0,0,11.07,0,0,10.8,0,0,14.04,0,0,13.5,
        ",
    );
}

#[test]
fn eltwise_dense_dense_dense() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_dense_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,
        ",
    );
}

#[test]
fn mult_dense_4d_tensors() {
    let output = Command::new(EX_DIR.to_owned() + "mult_dense_4d_tensors").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,
        ",
    );
}

#[test]
fn mult_dense_ij_ikj_kj() {
    let output = Command::new(EX_DIR.to_owned() + "mult_dense_ij-ikj-kj").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        21.76,21.76,21.76,21.76,21.76,21.76,21.76,21.76,21.76,21.76,21.76,21.76,21.76,21.76,21.76,21.76,
        ",
    );
}

#[test]
fn mult_dense_matrix_coo() {
    let output = Command::new(EX_DIR.to_owned() + "mult_dense_matrix_coo").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
        data =
        8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,
        ");
}

#[test]
fn mult_dense_matrix_csr() {
    let output = Command::new(EX_DIR.to_owned() + "mult_dense_matrix_csr").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
    data =
    8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,
    ");
}

#[test]
fn mult_dense_matrix_dcsr() {
    let output = Command::new(EX_DIR.to_owned() + "mult_dense_matrix_dcsr").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
    data =
    8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,
    ");
}

#[test]
fn mult_dense_matrix() {
    let output = Command::new(EX_DIR.to_owned() + "mult_dense_matrix").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data =
    29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,
    ",
    );
}

#[test]
fn mult_dense_matrix_vector() {
    let output = Command::new(EX_DIR.to_owned() + "mult_dense_matrix_vector").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data =
    136.16,136.16,136.16,136.16,136.16,136.16,136.16,136.16,
    ",
    );
}

#[test]
fn mult_dense_vector_coo() {
    let output = Command::new(EX_DIR.to_owned() + "mult_dense_vector_coo").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data =
    8.67,12.24,5.1,9.18,12.75,
    ",
    );
}

#[test]
fn mult_dense_vector_csr() {
    let output = Command::new(EX_DIR.to_owned() + "mult_dense_vector_csr").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data =
    8.67,12.24,5.1,9.18,12.75,
    ",
    );
}

#[test]
fn mult_dense_vector_dcsr() {
    let output = Command::new(EX_DIR.to_owned() + "mult_dense_vector_dcsr").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        8.67,12.24,5.1,9.18,12.75,
        ",
    );
}

#[test]
fn mult_mixed_i_i_ij() {
    let output = Command::new(EX_DIR.to_owned() + "mult_mixed_i-i-ij").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        4.08,7.65,5.1,13.77,17.34,
        ",
    );
}

// #[test]
// fn mult_spgemm_csr_csr_csr() {
//     let output = Command::new(EX_DIR.to_owned() + "mult_spgemm_csr_csr_csr").unwrap();
//     compare_strings(
//         &String::from_utf8_lossy(&output.stdout),
//         "
//         data =
//         5,
//         data =
//         0,
//         data =
//         0,2,4,5,7,9,
//         data =
//         0,3,1,4,2,0,3,1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//         data =
//         6.74,7,17,17.5,9,20.5,21.74,36.4,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//         ",
//     );
// }

#[test]
fn mult_spmm_coo_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mult_spmm_coo_dense").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
    data =
    4.08,4.08,4.08,4.08,7.65,7.65,7.65,7.65,5.1,5.1,5.1,5.1,13.77,13.77,13.77,13.77,17.34,17.34,17.34,17.34,
    ");
}

#[test]
fn mult_spmm_csr_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mult_spmm_csr_dense").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
    data =
    4.08,4.08,4.08,4.08,7.65,7.65,7.65,7.65,5.1,5.1,5.1,5.1,13.77,13.77,13.77,13.77,17.34,17.34,17.34,17.34,
    ");
}

#[test]
fn mult_spmm_dcsr_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mult_spmm_dcsr_dense").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
    data =
        4.08,4.08,4.08,4.08,7.65,7.65,7.65,7.65,5.1,5.1,5.1,5.1,13.77,13.77,13.77,13.77,17.34,17.34,17.34,17.34,
    ");
}

#[test]
fn mult_spmv_coo_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mult_spmv_coo_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data =
    4.08,7.65,5.1,13.77,17.34,
    ",
    );
}

#[test]
fn mult_spmv_csr_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mult_spmv_csr_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data =
    4.08,7.65,5.1,13.77,17.34,
    ",
    );
}

#[test]
fn mult_spmv_dcsr_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mult_spmv_dcsr_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data =
    4.08,7.65,5.1,13.77,17.34,
    ",
    );
}

// #[test]
// fn mult_ttm_csf_mode1(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttm_csf_mode1").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"
//     data =
//     0,1,2,3,
//     data =
//     3,1,6,
//     data =
//     0,1,2,3,
//     data =
//     2,3,5,
//     data =
//     4,
//     data =
//     0,
//     data =
//     3.51,3.51,3.51,3.51,5.697,5.697,5.697,5.697,8.1,8.1,8.1,8.1,
//     ");
// }

// #[test]
// fn mult_ttm_csf_mode2(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttm_csf_mode2").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"
//     data =
//     0,3,
//     data =
//     1,2,3,
//     data =
//     0,1,2,3,
//     data =
//     2,3,5,
//     data =
//     4,
//     data =
//     0,
//     data =
//     3.51,3.51,3.51,3.51,5.697,5.697,5.697,5.697,8.1,8.1,8.1,8.1,
//     ");
// }

// #[test]
// fn mult_ttm_csf_mode3(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttm_csf_mode3").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"
//     data =
//     0,3,
//     data =
//     1,2,3,
//     data =
//     0,1,2,3,
//     data =
//     3,1,6,
//     data =
//     4,
//     data =
//     0,
//     data =
//     3.51,3.51,3.51,3.51,5.697,5.697,5.697,5.697,8.1,8.1,8.1,8.1,
//     ");
// }

// #[test]
// fn mult_ttm_mg_mode1(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttm_mg_mode1").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"

//     ");
// }

// #[test]
// fn mult_ttm_mg_mode2(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttm_mg_mode2").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"

//     ");
// }

// #[test]
// fn mult_ttm_mg_mode3(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttm_mg_mode3").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"

//     ");
// }

// #[test]
// fn mult_ttv_csf_mode1(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttv_csf_mode1").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"
//     data =
//     0,1,2,3,
//     data =
//     3,1,6,
//     data =
//     0,1,2,3,
//     data =
//     2,3,5,
//     data =
//     4,
//     data =
//     0,
//     data =
//     3.51,3.51,3.51,3.51,5.697,5.697,5.697,5.697,8.1,8.1,8.1,8.1,
//     ");
// }

// #[test]
// fn mult_ttv_csf_mode2(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttv_csf_mode2").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"

//     ");
// }

// #[test]
// fn mult_ttv_csf_mode3(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttv_csf_mode3").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"
//     data =
//     0,3,
//     data =
//     1,2,3,
//     data =
//     0,1,2,3,
//     data =
//     3,1,6,
//     data =
//     4,
//     data =
//     0,
//     data =
//     3.51,3.51,3.51,3.51,5.697,5.697,5.697,5.697,8.1,8.1,8.1,8.1,
//     ");
// }

// #[test]
// fn mult_ttv_mg_mode1(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttv_mg_mode1").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"

//     ");
// }

// #[test]
// fn mult_ttv_mg_mode2(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttv_mg_mode2").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"

//     ");
// }

// #[test]
// fn mult_ttv_mg_mode3(){
//     let output = Command::new(EX_DIR.to_owned()+"mult_ttv_mg_mode3").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"

//     ");
// }

#[test]
fn multi_function() {
    let output = Command::new(EX_DIR.to_owned() + "multi_function").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,
    data =
    28.2,
    data =
    29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,
    ",
    );
}

// #[test]
// fn multi_mult_dense_matrix(){
//     let output = Command::new(EX_DIR.to_owned()+"multi_mult_dense_matrix").unwrap();
//     compare_strings(&String::from_utf8_lossy(&output.stdout),"

//     ");
// }

#[test]
fn sum_coo() {
    let output = Command::new(EX_DIR.to_owned() + "sum_coo").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data =
    28.2,
    ",
    );
}

#[test]
fn sum_csf() {
    let output = Command::new(EX_DIR.to_owned() + "sum_csf").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data =
    6.41,
    ",
    );
}

#[test]
fn sum_csr() {
    let output = Command::new(EX_DIR.to_owned() + "sum_csr").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data =
    28.2,
    ",
    );
}

#[test]
fn sum_dense_matrix() {
    let output = Command::new(EX_DIR.to_owned() + "sum_dense_matrix").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        59.2,
        ",
    );
}

#[test]
fn sum_dense_tensor() {
    let output = Command::new(EX_DIR.to_owned() + "sum_dense_tensor").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        236.8,
        ",
    );
}

#[test]
fn transpose_coo_matrix() {
    env::set_var("SORT_TYPE", "SEQ_QSORT");
    let output = Command::new(EX_DIR.to_owned() + "transpose_coo_matrix").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    0,9,
    data = 
    0,0,1,1,2,3,3,4,4,
    data = 
    -1,
    data = 
    0,3,1,4,2,0,3,1,4,
    data = 
    1,4.1,2,5.2,3,1.4,4,2.5,5,
    ",
    );
}

#[test]
fn transpose_coo_tensor() {
    env::set_var("SORT_TYPE", "SEQ_QSORT");
    let output = Command::new(EX_DIR.to_owned() + "transpose_coo_tensor").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    0,3,
    data = 
    1,3,6,
    data = 
    0,
    data = 
    2,1,3,
    data = 
    0,
    data =
    3,2,5,
    data =
    2.11,1.3,3,
    ",
    );
}

#[test]
fn transpose_csf_tensor() {
    env::set_var("SORT_TYPE", "SEQ_QSORT");
    let output = Command::new(EX_DIR.to_owned() + "transpose_csf_tensor").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    0,3,
    data = 
    2,3,5,
    data = 
    0,1,2,3,
    data = 
    1,2,3,
    data = 
    0,1,2,3,
    data =
    3,1,6,
    data =
    1.3,2.11,3,
    ",
    );
}

#[test]
fn transpose_csr_matrix() {
    env::set_var("SORT_TYPE", "SEQ_QSORT");
    let output = Command::new(EX_DIR.to_owned() + "transpose_csr_matrix").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    5,
    data = 
    -1,
    data = 
    0,2,4,5,7,9,
    data = 
    0,3,1,4,2,0,3,1,4,
    data = 
    1,4.1,2,5.2,3,1.4,4,2.5,5,
    ",
    );
}

#[test]
fn transpose_dense_matrix() {
    let output = Command::new(EX_DIR.to_owned() + "transpose_dense_matrix").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        3.2,3.2,3.2,3.2,3.2,3.2,3.2,3.2,3.2,3.2,3.2,3.2,3.2,3.2,3.2,3.2,
        ",
    );
}

#[test]
fn transpose_dense_tensor() {
    let output = Command::new(EX_DIR.to_owned() + "transpose_dense_tensor").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data =
        3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,3.7,
        ",
    );
}

#[test]
fn utility_get_time() {
    let _output = Command::new(EX_DIR.to_owned() + "utility_get_time").unwrap();
}

 #[test]
 fn utility_print_coo_multi() {
     let output = Command::new(EX_DIR.to_owned() + "utility_print_coo_multi").unwrap();
     compare_strings(
         &String::from_utf8_lossy(&output.stdout),
         "
     data = 
     0,7,
     data = 
     0,0,1,1,3,4,4,
     data = 
     -1,
     data = 
     0,3,0,1,1,2,3,
     data = 
     1,2,3,4,5,6,7,
     data = 
     0,9,
     data = 
     0,0,1,1,2,3,3,4,4,
     data = 
     -1,
     data = 
     0,3,1,4,2,0,3,1,4,
     data = 
     1,1.4,2,2.5,3,4.1,4,5.2,5,
     ",
     );
 }

#[test]
fn utility_print_coo() {
    let output = Command::new(EX_DIR.to_owned() + "utility_print_coo").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    0,7,
    data = 
    0,0,1,1,3,4,4,
    data = 
    -1,
    data = 
    0,3,0,1,1,2,3,
    data = 
    1,2,3,4,5,6,7,
    ",
    );
}

 #[test]
 fn utility_print_csf_multi() {
     let output = Command::new(EX_DIR.to_owned() + "utility_print_csf_multi").unwrap();
     compare_strings(
         &String::from_utf8_lossy(&output.stdout),
         "
         data = 
         0,3,
         data = 
         1,2,3,
         data = 
         0,1,2,3,
         data = 
         3,1,6,
         data = 
         0,1,2,3,
         data = 
         2,3,5,
         data = 
         1.3,2.11,3,
         data = 
         0,3,
         data = 
         1,2,3,
         data = 
         0,1,2,3,
         data = 
         3,1,6,
         data = 
         0,1,2,3,
         data = 
         2,3,5,
         data = 
         1.3,2.11,3,
     ",
     );
 }

 #[test]
 fn utility_print_csr_multi() {
     let output = Command::new(EX_DIR.to_owned() + "utility_print_csr_multi").unwrap();
     compare_strings(
         &String::from_utf8_lossy(&output.stdout),
         "
         data = 
         5,
         data = 
         -1,
         data = 
        0,2,4,4,5,7,
        data = 
        0,3,0,1,1,2,3,
        data = 
        1,2,3,4,5,6,7,
        data = 
        5,
        data = 
        -1,
        data = 
        0,2,4,5,7,9,
        data = 
        0,3,1,4,2,0,3,1,4,
        data = 
        1,1.4,2,2.5,3,4.1,4,5.2,5,
        data = 
        5,
        data = 
        -1,
        data = 
        0,2,4,4,5,7,
        data = 
        0,3,0,1,1,2,3,
        data = 
        1,2,3,4,5,6,7,
    ",
    );
}

#[test]
fn utility_print_csr() {
    let output = Command::new(EX_DIR.to_owned() + "utility_print_csr").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data = 
        5,
        data = 
        -1,
        data = 
        0,2,4,4,5,7,
        data = 
        0,3,0,1,1,2,3,
        data = 
        1,2,3,4,5,6,7,
        ",
    );
}

#[test]
fn utility_print_dense() {
    let output = Command::new(EX_DIR.to_owned() + "utility_print_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data = 
        2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,2.3,
        ",
    );
}
