use assert_cmd::Command;

const EX_DIR: &str = "./target/debug/examples/";

fn remove_whitespace(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}
fn compare_strings(a: &str, b: &str) {
    let a = remove_whitespace(a);
    let b = remove_whitespace(b);
    assert_eq!(a, b);
}

#[test]
fn dtranspose_eltwise_dense_dense() {
    let output = Command::new(EX_DIR.to_owned() + "dtranspose_eltwise_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,6.4,
    ",
    );
}

#[test]
fn dtranspose_mult_dense_dense() {
    let output = Command::new(EX_DIR.to_owned() + "dtranspose_mult_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    12.8,12.8,12.8,12.8,12.8,12.8,12.8,12.8,12.8,12.8,12.8,12.8,12.8,12.8,12.8,12.8,
    ",
    );
}

#[test]
fn sp_transpose_csr_eltwise_csr_csr() {
    let output = Command::new(EX_DIR.to_owned() + "sp_transpose_csr_eltwise_csr_csr").unwrap();
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
    0,3,1,4,2,0,3,1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    data = 
    1,5.74,4,13,9,5.74,16,13,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    ",
    );
}

#[test]
fn sp_transpose_csr_eltwise_csr_dense() {
    let output = Command::new(EX_DIR.to_owned() + "sp_transpose_csr_eltwise_csr_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
        data=
        2.3,6.9,0,0,0,0,9.2,0,11.5,0,0,0,0,0,13.8,4.6,0,0,0,16.1,
        ",
    );
}
