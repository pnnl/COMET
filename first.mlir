module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = "ta.static_index_label"(%c0, %c4, %c1) : (index, index, index) -> !ta.range
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %2 = "ta.dynamic_index_label"(%c0_2, %c1_3) : (index, index) -> !ta.range
    %3 = "ta.sparse_tensor_decl"(%1, %2) {format = "CSR", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
    %4 = "ta.dense_tensor_decl"(%0, %1) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<4x?xf64>
    %5 = "ta.dense_tensor_decl"(%0, %2) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<4x?xf64>
    "ta.fill"(%4) {value = 1.000000e+00 : f64} : (tensor<4x?xf64>) -> ()
    "ta.fill_from_file"(%3) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
    "ta.fill"(%5) {value = 0.000000e+00 : f64} : (tensor<4x?xf64>) -> ()
    %6 = "it.ComputeRHS"(%4, %3) {allBlocks = [["UNK", "UNK"], ["UNK", "UNK"]], allFormats = [["D", "D"], ["D", "CU"]], allPerms = [[0, 1], [1, 2]]} : (tensor<4x?xf64>, tensor<?x?xf64>) -> tensor<*xf64>
    %7 = "it.ComputeLHS"(%5) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[0, 2]]} : (tensor<4x?xf64>) -> tensor<*xf64>
    %8 = "it.Compute"(%6, %7) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
    %9 = "it.Indices"(%8) {indices = [2]} : (i64) -> i64
    %10 = "it.Indices"(%9) {indices = [1]} : (i64) -> i64
    %11 = "it.Indices"(%10) {indices = [0]} : (i64) -> i64
    %12 = "it.itree"(%11) : (i64) -> i64
    "ta.print"(%5) : (tensor<4x?xf64>) -> ()
    "ta.print"(%3) : (tensor<?x?xf64>) -> ()
    return
  }
}
