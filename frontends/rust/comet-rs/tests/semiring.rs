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
fn eltwise_monoid_min_dense_dense_dense() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_monoid_min_dense_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,
    ",
    );
}

#[test]
fn eltwise_monoid_plus_coo_dense_coo() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_monoid_plus_coo_dense_coo").unwrap();
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
    3.7,4.1,4.7,5.2,5.7,6.8,6.7,7.9,7.7,
    ",
    );
}

#[test]
fn eltwise_monoid_plus_dense_dense_dense() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_monoid_plus_dense_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    5.9,5.9,5.9,5.9,5.9,5.9,5.9,5.9,5.9,5.9,5.9,5.9,5.9,5.9,5.9,5.9,
    ",
    );
}

#[test]
fn eltwise_monoid_times_coo_dense_coo() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_monoid_times_coo_dense_coo").unwrap();
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

#[test]
fn eltwise_monoid_times_dense_4d_tensors() {
    let output = Command::new(EX_DIR.to_owned() + "eltwise_monoid_times_dense_4d_tensors").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,7.92,
    ",
    );
}

#[test]
fn eltwise_monoid_times_dense_dense_dense() {
    let output =
        Command::new(EX_DIR.to_owned() + "eltwise_monoid_times_dense_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,
    ",
    );
}

// #[test]
// fn mm_semiring_anypair_csr_csr_csr() {
//     let output = Command::new(EX_DIR.to_owned() + "mm_semiring_anypair_csr_csr_csr").unwrap();
//     compare_strings(
//         &String::from_utf8_lossy(&output.stdout),
//         "
//     data = 
//     5,
//     data = 
//     0,
//     data = 
//     0,2,4,5,7,9,
//     data = 
//     0,3,1,4,2,0,3,1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     data = 
//     1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     ",
//     );
// }

// #[test]
// fn mm_semiring_minfirst_csr_csr_csr() {
//     let output = Command::new(EX_DIR.to_owned() + "mm_semiring_minfirst_csr_csr_csr").unwrap();
//     compare_strings(
//         &String::from_utf8_lossy(&output.stdout),
//         "
//     data = 
//     5,
//     data = 
//     0,
//     data = 
//     0,2,4,5,7,9,
//     data = 
//     0,3,1,4,2,0,3,1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     data = 
//     1,1,2,2,3,4,4,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     ",
//     );
// }

// #[test]
// fn mm_semiring_minplus_csr_csr_csr() {
//     let output = Command::new(EX_DIR.to_owned() + "mm_semiring_minplus_csr_csr_csr").unwrap();
//     compare_strings(
//         &String::from_utf8_lossy(&output.stdout),
//         "
//     data = 
//     5,
//     data = 
//     0,
//     data = 
//     0,2,4,5,7,9,
//     data = 
//     0,3,1,4,2,0,3,1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     data = 
//     2,2.4,4,4.5,6,5.1,5.5,7.2,7.7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     ",
//     );
// }

// #[test]
// fn mm_semiring_minsecond_csr_csr_csr() {
//     let output = Command::new(EX_DIR.to_owned() + "mm_semiring_minsecond_csr_csr_csr").unwrap();
//     compare_strings(
//         &String::from_utf8_lossy(&output.stdout),
//         "
//     data = 
//     5,
//     data = 
//     0,
//     data = 
//     0,2,4,5,7,9,
//     data = 
//     0,3,1,4,2,0,3,1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     data = 
//     1,1.4,2,2.5,3,1,1.4,2,2.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     ",
//     );
// }

// #[test]
// fn mm_semiring_plusfirst_csr_csr_csr() {
//     let output = Command::new(EX_DIR.to_owned() + "mm_semiring_plusfirst_csr_csr_csr").unwrap();
//     compare_strings(
//         &String::from_utf8_lossy(&output.stdout),
//         "
//     data = 
//     5,
//     data = 
//     0,
//     data = 
//     0,2,4,5,7,9,
//     data = 
//     0,3,1,4,2,0,3,1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     data = 
//     2.4,2.4,4.5,4.5,3,8.1,8.1,10.2,10.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     ",
//     );
// }

// #[test]
// fn mm_semiring_pluspair_csr_csr_csr() {
//     let output = Command::new(EX_DIR.to_owned() + "mm_semiring_pluspair_csr_csr_csr").unwrap();
//     compare_strings(
//         &String::from_utf8_lossy(&output.stdout),
//         "
//     data = 
//     5,
//     data = 
//     0,
//     data = 
//     0,2,4,5,7,9,
//     data = 
//     0,3,1,4,2,0,3,1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     data = 
//     2,2,2,2,1,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     ",
//     );
// }

// #[test]
// fn mm_semiring_plussecond_csr_csr_csr() {
//     let output = Command::new(EX_DIR.to_owned() + "mm_semiring_plussecond_csr_csr_csr").unwrap();
//     compare_strings(
//         &String::from_utf8_lossy(&output.stdout),
//         "
//     data = 
//     5,
//     data = 
//     0,
//     data = 
//     0,2,4,5,7,9,
//     data = 
//     0,3,1,4,2,0,3,1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     data = 
//     5.1,5.4,7.2,7.5,3,5.1,5.4,7.2,7.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     ",
//     );
// }

#[test]
fn mm_semiring_plustimes_coo_dense_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mm_semiring_plustimes_coo_dense_dense").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
    data = 
    4.08,4.08,4.08,4.08,7.65,7.65,7.65,7.65,5.1,5.1,5.1,5.1,13.77,13.77,13.77,13.77,17.34,17.34,17.34,17.34,
    ");
}

// #[test]
// fn mm_semiring_plustimes_csr_csr_csr() {
//     let output = Command::new(EX_DIR.to_owned() + "mm_semiring_plustimes_csr_csr_csr").unwrap();
//     compare_strings(
//         &String::from_utf8_lossy(&output.stdout),
//         "
//     data = 
//     5,
//     data = 
//     0,
//     data = 
//     0,2,4,5,7,9,
//     data = 
//     0,3,1,4,2,0,3,1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     data = 
//     6.74,7,17,17.5,9,20.5,21.74,36.4,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     ",
//     );
// }

#[test]
fn mm_semiring_plustimes_csr_dense_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mm_semiring_plustimes_csr_dense_dense").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
    data = 
    4.08,4.08,4.08,4.08,7.65,7.65,7.65,7.65,5.1,5.1,5.1,5.1,13.77,13.77,13.77,13.77,17.34,17.34,17.34,17.34,
    ");
}

#[test]
fn mm_semiring_plustimes_dcsr_dense_dense() {
    let output =
        Command::new(EX_DIR.to_owned() + "mm_semiring_plustimes_dcsr_dense_dense").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
    data = 
    4.08,4.08,4.08,4.08,7.65,7.65,7.65,7.65,5.1,5.1,5.1,5.1,13.77,13.77,13.77,13.77,17.34,17.34,17.34,17.34,
    ");
}

#[test]
fn mm_semiring_plustimes_dense_4d_tensors() {
    let output =
        Command::new(EX_DIR.to_owned() + "mm_semiring_plustimes_dense_4d_tensors").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,
    ",
    );
}

#[test]
fn mm_semiring_plustimes_dense_coo_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mm_semiring_plustimes_dense_coo_dense").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
    data = 
    8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,
    ");
}

#[test]
fn mm_semiring_plustimes_dense_csr_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mm_semiring_plustimes_dense_csr_dense").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
    data = 
    8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,
    ");
}

#[test]
fn mm_semiring_plustimes_dense_dcsr_dense() {
    let output =
        Command::new(EX_DIR.to_owned() + "mm_semiring_plustimes_dense_dcsr_dense").unwrap();
    compare_strings(&String::from_utf8_lossy(&output.stdout),"
    data = 
    8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,8.67,12.24,5.1,9.18,12.75,
    ");
}

#[test]
fn mm_semiring_plustimes_dense_dense_dense() {
    let output =
        Command::new(EX_DIR.to_owned() + "mm_semiring_plustimes_dense_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,
    ",
    );
}

#[test]
fn mv_semiring_plustimes_coo_dense_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mv_semiring_plustimes_coo_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    4.08,7.65,5.1,13.77,17.34,
    ",
    );
}

#[test]
fn mv_semiring_plustimes_csr_dense_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mv_semiring_plustimes_csr_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    4.08,7.65,5.1,13.77,17.34,
    ",
    );
}

#[test]
fn mv_semiring_plustimes_dcsr_dense_dense() {
    let output =
        Command::new(EX_DIR.to_owned() + "mv_semiring_plustimes_dcsr_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    4.08,7.65,5.1,13.77,17.34,
    ",
    );
}

#[test]
fn mv_semiring_plustimes_dense_coo_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mv_semiring_plustimes_dense_coo_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    8.67,12.24,5.1,9.18,12.75,
    ",
    );
}

#[test]
fn mv_semiring_plustimes_dense_csr_dense() {
    let output = Command::new(EX_DIR.to_owned() + "mv_semiring_plustimes_dense_csr_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    8.67,12.24,5.1,9.18,12.75,
    ",
    );
}

#[test]
fn mv_semiring_plustimes_dense_dcsr_dense() {
    let output =
        Command::new(EX_DIR.to_owned() + "mv_semiring_plustimes_dense_dcsr_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    8.67,12.24,5.1,9.18,12.75,
    ",
    );
}

#[test]
fn mv_semiring_plustimes_dense_dense_dense() {
    let output =
        Command::new(EX_DIR.to_owned() + "mv_semiring_plustimes_dense_dense_dense").unwrap();
    compare_strings(
        &String::from_utf8_lossy(&output.stdout),
        "
    data = 
    136.16,136.16,136.16,136.16,136.16,136.16,136.16,136.16,
    ",
    );
}
