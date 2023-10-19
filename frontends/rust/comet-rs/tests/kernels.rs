use assert_cmd::Command;

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
fn ccsd_t1_21_loops() {
    let output = Command::new(EX_DIR.to_owned() + "ccsd_t1_21_loops").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    250.24,250.24,250.24,250.24,250.24,250.24,250.24,250.24,
    ",
    );
}

#[test]
fn ccsd_t1_3_loops() {
    let output = Command::new(EX_DIR.to_owned() + "ccsd_t1_3_loops").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    15.64,15.64,15.64,15.64,15.64,15.64,15.64,15.64,
    ",
    );
}

#[test]
fn ccsd_t1_4_loops() {
    let output = Command::new(EX_DIR.to_owned() + "ccsd_t1_4_loops").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    62.56,62.56,62.56,62.56,62.56,62.56,62.56,62.56,
    ",
    );
}
