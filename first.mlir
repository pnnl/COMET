// -----// IR Dump After (anonymous namespace)::FuncOpLoweringPass () //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
    %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "ELL", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
    %3 = "ta.dense_tensor_decl"(%1) {format = "Dense"} : (!ta.range) -> tensor<?xf64>
    %4 = "ta.dense_tensor_decl"(%0) {format = "Dense"} : (!ta.range) -> tensor<?xf64>
    "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
    "ta.fill"(%3) {value = 1.700000e+00 : f64} : (tensor<?xf64>) -> ()
    "ta.fill"(%4) {value = 0.000000e+00 : f64} : (tensor<?xf64>) -> ()
    %5 = "ta.mul"(%2, %3, %0) {__alpha__ = 1.000000e+00 : f64, __beta__ = 0.000000e+00 : f64, formats = ["ELL", "Dense", "Dense"], indexing_maps = [#map, #map1, #map2], semiring = "plusxy_times"} : (tensor<?x?xf64>, tensor<?xf64>, !ta.range) -> tensor<?xf64>
    "ta.set_op"(%5, %4) {__beta__ = 0.000000e+00 : f64} : (tensor<?xf64>, tensor<?xf64>) -> ()
    "ta.print"(%4) : (tensor<?xf64>) -> ()
    return
  }
}


// -----// IR Dump After (anonymous namespace)::RemoveLabeledTensorOpPass () //----- //
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
  %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "ELL", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %3 = "ta.dense_tensor_decl"(%1) {format = "Dense"} : (!ta.range) -> tensor<?xf64>
  %4 = "ta.dense_tensor_decl"(%0) {format = "Dense"} : (!ta.range) -> tensor<?xf64>
  "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
  "ta.fill"(%3) {value = 1.700000e+00 : f64} : (tensor<?xf64>) -> ()
  "ta.fill"(%4) {value = 0.000000e+00 : f64} : (tensor<?xf64>) -> ()
  %5 = "ta.mul"(%2, %3, %0) {__alpha__ = 1.000000e+00 : f64, __beta__ = 0.000000e+00 : f64, formats = ["ELL", "Dense", "Dense"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], semiring = "plusxy_times"} : (tensor<?x?xf64>, tensor<?xf64>, !ta.range) -> tensor<?xf64>
  "ta.set_op"(%5, %4) {__beta__ = 0.000000e+00 : f64} : (tensor<?xf64>, tensor<?xf64>) -> ()
  "ta.print"(%4) : (tensor<?xf64>) -> ()
  return
}

// -----// IR Dump After (anonymous namespace)::LowerTAMulChainPass () //----- //
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
  %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "ELL", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %3 = "ta.dense_tensor_decl"(%1) {format = "Dense"} : (!ta.range) -> tensor<?xf64>
  %4 = "ta.dense_tensor_decl"(%0) {format = "Dense"} : (!ta.range) -> tensor<?xf64>
  "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
  "ta.fill"(%3) {value = 1.700000e+00 : f64} : (tensor<?xf64>) -> ()
  "ta.fill"(%4) {value = 0.000000e+00 : f64} : (tensor<?xf64>) -> ()
  %5 = "ta.mul"(%2, %3, %0) {__alpha__ = 1.000000e+00 : f64, __beta__ = 0.000000e+00 : f64, formats = ["ELL", "Dense", "Dense"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], semiring = "plusxy_times"} : (tensor<?x?xf64>, tensor<?xf64>, !ta.range) -> tensor<?xf64>
  "ta.set_op"(%5, %4) {__beta__ = 0.000000e+00 : f64} : (tensor<?xf64>, tensor<?xf64>) -> ()
  "ta.print"(%4) : (tensor<?xf64>) -> ()
  return
}

// -----// IR Dump After (anonymous namespace)::TensorAlgebraCheckImplicitTensorDeclPass () //----- //
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
  %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "ELL", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %3 = "ta.dense_tensor_decl"(%1) {format = "Dense"} : (!ta.range) -> tensor<?xf64>
  %4 = "ta.dense_tensor_decl"(%0) {format = "Dense"} : (!ta.range) -> tensor<?xf64>
  "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
  "ta.fill"(%3) {value = 1.700000e+00 : f64} : (tensor<?xf64>) -> ()
  "ta.fill"(%4) {value = 0.000000e+00 : f64} : (tensor<?xf64>) -> ()
  %5 = "ta.mul"(%2, %3, %0) {__alpha__ = 1.000000e+00 : f64, __beta__ = 0.000000e+00 : f64, formats = ["ELL", "Dense", "Dense"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>], semiring = "plusxy_times"} : (tensor<?x?xf64>, tensor<?xf64>, !ta.range) -> tensor<?xf64>
  "ta.set_op"(%5, %4) {__beta__ = 0.000000e+00 : f64} : (tensor<?xf64>, tensor<?xf64>) -> ()
  "ta.print"(%4) : (tensor<?xf64>) -> ()
  return
}

Assertion failed: (format.size() == indices.size()), function Tensor, file Tensor.h, line 66.
