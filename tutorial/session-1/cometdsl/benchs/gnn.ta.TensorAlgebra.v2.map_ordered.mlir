// -----// IR Dump After (anonymous namespace)::FuncOpLoweringPass () //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  func.func @main() {
    %0 = "ta.index_label"() : () -> !ta.index
    %1 = "ta.index_label"() : () -> !ta.index
    %2 = "ta.index_label"() : () -> !ta.index
    %3 = "ta.index_label"() : () -> !ta.index
    %4 = "ta.spTensor_decl"() <{format = "CSR", temporal_tensor = false}> : () -> !ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>
    %c0 = arith.constant 0 : index
    %5 = "ta.dim"(%4, %c0) : (!ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>, index) -> index
    %c1 = arith.constant 1 : index
    %6 = "ta.dim"(%4, %c1) : (!ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>, index) -> index
    %7 = "ta.dense_tensor_decl"(%6) <{format = "Dense"}> : (index) -> tensor<?x16xf64>
    %8 = "ta.dense_tensor_decl"() <{format = "Dense"}> : () -> tensor<16x16xf64>
    %9 = "ta.dense_tensor_decl"(%5) <{format = "Dense"}> : (index) -> tensor<?x16xf64>
    %10 = "ta.dense_tensor_decl"(%5) <{format = "Dense"}> : (index) -> tensor<?x16xf64>
    "ta.fill_from_file"(%4) <{filename = "SPARSE_FILE_NAME0", readMode = 1 : i32}> : (!ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>) -> ()
    "ta.fill"(%7) <{value = 1.200000e+00 : f64}> : (tensor<?x16xf64>) -> ()
    "ta.fill"(%8) <{value = 3.400000e+00 : f64}> : (tensor<16x16xf64>) -> ()
    "ta.fill"(%9) <{value = 0.000000e+00 : f64}> : (tensor<?x16xf64>) -> ()
    "ta.fill"(%10) <{value = 0.000000e+00 : f64}> : (tensor<?x16xf64>) -> ()
    %11 = "ta.getTime"() : () -> f64
    %12 = "ta.mul"(%4, %7, %0, %1, %1, %3, %0, %3) <{MaskType = "none", formats = ["CSR", "Dense", "Dense"], indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 1, 1, 6, 0>, semiring = "plusxy_times"}> {__alpha__ = 1.000000e+00 : f64, __beta__ = 0.000000e+00 : f64} : (!ta.sparse_tensor<f64, i64, ?x?, d, unk, cu, unk>, tensor<?x16xf64>, !ta.index, !ta.index, !ta.index, !ta.index, !ta.index, !ta.index) -> tensor<?x16xf64>
    "ta.set_op"(%12, %10) {__beta__ = 0.000000e+00 : f64} : (tensor<?x16xf64>, tensor<?x16xf64>) -> ()
    %13 = "ta.mul"(%10, %8, %0, %3, %3, %2, %0, %2) <{MaskType = "none", formats = ["Dense", "Dense", "Dense"], indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 1, 1, 6, 0>, semiring = "plusxy_times"}> {__alpha__ = 1.000000e+00 : f64, __beta__ = 0.000000e+00 : f64} : (tensor<?x16xf64>, tensor<16x16xf64>, !ta.index, !ta.index, !ta.index, !ta.index, !ta.index, !ta.index) -> tensor<?x16xf64>
    "ta.set_op"(%13, %9) {__beta__ = 0.000000e+00 : f64} : (tensor<?x16xf64>, tensor<?x16xf64>) -> ()
    %14 = "ta.getTime"() : () -> f64
    %15 = "ta.reduce"(%9) : (tensor<?x16xf64>) -> f64
    "ta.print"(%15) : (f64) -> ()
    "ta.print_elapsed_time"(%11, %14) : (f64, f64) -> ()
    return
  }
}