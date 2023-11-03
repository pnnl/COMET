// -----// IR Dump After {anonymous}::FuncOpLoweringPass () //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
    %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "CSR", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
    %3 = "ta.dense_tensor_decl"(%1, %0) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
    %4 = "ta.dense_tensor_decl"(%1, %0) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
    "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
    "ta.fill"(%3) {value = 1.000000e+00 : f64} : (tensor<?x?xf64>) -> ()
    %5 = "ta.mul"(%2, %3, %1, %0) {MaskType = "none", __alpha__ = 1.000000e+00 : f64, __beta__ = 0.000000e+00 : f64, formats = ["CSR", "Dense", "Dense"], indexing_maps = [#map, #map1, #map1], operand_segment_sizes = array<i32: 1, 1, 2, 0>, semiring = "plusxy_times"} : (tensor<?x?xf64>, tensor<?x?xf64>, !ta.range, !ta.range) -> tensor<?x?xf64>
    "ta.set_op"(%5, %4) {__beta__ = 0.000000e+00 : f64} : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
    "ta.print"(%2) : (tensor<?x?xf64>) -> ()
    "ta.print"(%3) : (tensor<?x?xf64>) -> ()
    "ta.print"(%4) : (tensor<?x?xf64>) -> ()
    return
  }
}


// -----// IR Dump After {anonymous}::RemoveLabeledTensorOpPass () //----- //
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
  %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "CSR", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %3 = "ta.dense_tensor_decl"(%1, %0) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %4 = "ta.dense_tensor_decl"(%1, %0) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
  "ta.fill"(%3) {value = 1.000000e+00 : f64} : (tensor<?x?xf64>) -> ()
  %5 = "ta.mul"(%2, %3, %1, %0) {MaskType = "none", __alpha__ = 1.000000e+00 : f64, __beta__ = 0.000000e+00 : f64, formats = ["CSR", "Dense", "Dense"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>], operand_segment_sizes = array<i32: 1, 1, 2, 0>, semiring = "plusxy_times"} : (tensor<?x?xf64>, tensor<?x?xf64>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%5, %4) {__beta__ = 0.000000e+00 : f64} : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
  "ta.print"(%2) : (tensor<?x?xf64>) -> ()
  "ta.print"(%3) : (tensor<?x?xf64>) -> ()
  "ta.print"(%4) : (tensor<?x?xf64>) -> ()
  return
}

// -----// IR Dump After {anonymous}::LowerTAMulChainPass () //----- //
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
  %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "CSR", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %3 = "ta.dense_tensor_decl"(%1, %0) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %4 = "ta.dense_tensor_decl"(%1, %0) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
  "ta.fill"(%3) {value = 1.000000e+00 : f64} : (tensor<?x?xf64>) -> ()
  %5 = "ta.mul"(%2, %3, %1, %0) {MaskType = "none", __alpha__ = 1.000000e+00 : f64, __beta__ = 0.000000e+00 : f64, formats = ["CSR", "Dense", "Dense"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>], operand_segment_sizes = array<i32: 1, 1, 2, 0>, semiring = "plusxy_times"} : (tensor<?x?xf64>, tensor<?x?xf64>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%5, %4) {__beta__ = 0.000000e+00 : f64} : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
  "ta.print"(%2) : (tensor<?x?xf64>) -> ()
  "ta.print"(%3) : (tensor<?x?xf64>) -> ()
  "ta.print"(%4) : (tensor<?x?xf64>) -> ()
  return
}

// -----// IR Dump After {anonymous}::TensorAlgebraCheckImplicitTensorDeclPass () //----- //
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
  %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "CSR", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %3 = "ta.dense_tensor_decl"(%1, %0) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %4 = "ta.dense_tensor_decl"(%1, %0) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
  "ta.fill"(%3) {value = 1.000000e+00 : f64} : (tensor<?x?xf64>) -> ()
  %5 = "ta.mul"(%2, %3, %1, %0) {MaskType = "none", __alpha__ = 1.000000e+00 : f64, __beta__ = 0.000000e+00 : f64, formats = ["CSR", "Dense", "Dense"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>], operand_segment_sizes = array<i32: 1, 1, 2, 0>, semiring = "plusxy_times"} : (tensor<?x?xf64>, tensor<?x?xf64>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%5, %4) {__beta__ = 0.000000e+00 : f64} : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
  "ta.print"(%2) : (tensor<?x?xf64>) -> ()
  "ta.print"(%3) : (tensor<?x?xf64>) -> ()
  "ta.print"(%4) : (tensor<?x?xf64>) -> ()
  return
}

// -----// IR Dump After {anonymous}::LowerTensorAlgebraToIndexTreePass () //----- //
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
  %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "CSR", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %3 = "ta.dense_tensor_decl"(%1, %0) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %4 = "ta.dense_tensor_decl"(%1, %0) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
  "ta.fill"(%3) {value = 1.000000e+00 : f64} : (tensor<?x?xf64>) -> ()
  %5 = "it.ComputeRHS"(%2, %3) {allBlocks = [["UNK", "UNK"], ["UNK", "UNK"]], allFormats = [["D", "CU"], ["D", "D"]], allPerms = [[0, 1], [1, 0]]} : (tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<*xf64>
  %6 = "it.ComputeLHS"(%4) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[1, 0]]} : (tensor<?x?xf64>) -> tensor<*xf64>
  %7 = "it.Compute"(%5, %6) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
  %8 = "it.Indices"(%7) {indices = [1]} : (i64) -> i64
  %9 = "it.Indices"(%8) {indices = [0]} : (i64) -> i64
  %10 = "it.itree"(%9) : (i64) -> i64
  "ta.print"(%2) : (tensor<?x?xf64>) -> ()
  "ta.print"(%3) : (tensor<?x?xf64>) -> ()
  "ta.print"(%4) : (tensor<?x?xf64>) -> ()
  return
}

/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1144  SparseInputTensorDeclOpLowering in format begin
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1145 %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "CSR", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1156  --- CSR
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1158  2
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1168 
Check the tensor is input or output
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1169 %5 = "it.ComputeRHS"(%2, %3) {allBlocks = [["UNK", "UNK"], ["UNK", "UNK"]], allFormats = [["D", "CU"], ["D", "D"]], allPerms = [[0, 1], [1, 0]]} : (tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<*xf64>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1245  used in ta.itComputeRHS op
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1168 
Check the tensor is input or output
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1169 "ta.print"(%2) : (tensor<?x?xf64>) -> ()
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1260  the tensor is in PrintOp
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1168 
Check the tensor is input or output
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1169 "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1256  the tensor is in fill_from_file op
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1287  isOutputTensor: 0
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1296  CSR isDense: 0
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1301  Sparse input tensor 
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1356  filename: SPARSE_FILE_NAME0
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1360  SPARSE_FILE_NAME0
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1363  readMode: 1
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1367 sp_decl.getParameterCount(): 13
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1370  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1371 %alloc = memref.alloc() : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1376  Get the dim_format
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1395  Parsed fileID: 0
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1413  2D
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:291 Inserting insertReadFileLibCall
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:311  Rank Size is 2
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:346 Adding read_input_2D_f64 to the module
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:373 Adding read_input_sizes_2D_f64 to the module
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %2 = memref.load %alloc[%c0_4] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %3 = memref.load %alloc[%c1_5] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %4 = memref.load %alloc[%c2_6] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %5 = memref.load %alloc[%c3_7] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %6 = memref.load %alloc[%c4] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %7 = memref.load %alloc[%c5] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %8 = memref.load %alloc[%c6] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %9 = memref.load %alloc[%c7] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %10 = memref.load %alloc[%c8] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %11 = memref.load %alloc[%c9] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %12 = memref.load %alloc[%c10] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %13 = memref.load %alloc[%c11] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1466  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1467 %14 = memref.load %alloc[%c12] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1477 %2 = memref.load %alloc[%c0_4] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1479  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1480 %alloc_8 = memref.alloc(%2) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1477 %3 = memref.load %alloc[%c1_5] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1479  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1480 %alloc_13 = memref.alloc(%3) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1477 %4 = memref.load %alloc[%c2_6] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1479  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1480 %alloc_18 = memref.alloc(%4) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1477 %5 = memref.load %alloc[%c3_7] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1479  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1480 %alloc_23 = memref.alloc(%5) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1477 %6 = memref.load %alloc[%c4] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1479  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1480 %alloc_28 = memref.alloc(%6) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1477 %7 = memref.load %alloc[%c5] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1479  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1480 %alloc_33 = memref.alloc(%7) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1477 %8 = memref.load %alloc[%c6] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1479  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1480 %alloc_38 = memref.alloc(%8) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1477 %9 = memref.load %alloc[%c7] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1479  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1480 %alloc_43 = memref.alloc(%9) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1492  /home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1493 %alloc_48 = memref.alloc(%10) : memref<?xf64>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1556  Generate read_input_2D or read_input_3D functions
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1603 SparseTensorConstructOp generated for input sparse tensor:
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1604 %24 = ta.sptensor_construct(%15, %16, %17, %18, %19, %20, %21, %22, %23, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1612 %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1613 %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1622 %25 = "ta.static_index_label"(%c0, %11, %c1) : (index, index, index) -> !ta.range
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1612 %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1613 %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1622 %26 = "ta.static_index_label"(%c0_0, %12, %c1_1) : (index, index, index) -> !ta.range
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1668  SparseInputTensorDeclOpLowering in format end
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1803 ---------------SparseTensorDeclLoweringPass end
// -----// IR Dump After {anonymous}::SparseTensorDeclLoweringPass () //----- //
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
  %alloc = memref.alloc() : memref<13xindex>
  %cast = memref.cast %alloc : memref<13xindex> to memref<*xindex>
  %c-1 = arith.constant -1 : index
  %c0_2 = arith.constant 0 : index
  %c1_3 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  call @read_input_sizes_2D_f64(%c0_i32, %c0_2, %c-1, %c1_3, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
  %c0_4 = arith.constant 0 : index
  %2 = memref.load %alloc[%c0_4] : memref<13xindex>
  %c1_5 = arith.constant 1 : index
  %3 = memref.load %alloc[%c1_5] : memref<13xindex>
  %c2_6 = arith.constant 2 : index
  %4 = memref.load %alloc[%c2_6] : memref<13xindex>
  %c3_7 = arith.constant 3 : index
  %5 = memref.load %alloc[%c3_7] : memref<13xindex>
  %c4 = arith.constant 4 : index
  %6 = memref.load %alloc[%c4] : memref<13xindex>
  %c5 = arith.constant 5 : index
  %7 = memref.load %alloc[%c5] : memref<13xindex>
  %c6 = arith.constant 6 : index
  %8 = memref.load %alloc[%c6] : memref<13xindex>
  %c7 = arith.constant 7 : index
  %9 = memref.load %alloc[%c7] : memref<13xindex>
  %c8 = arith.constant 8 : index
  %10 = memref.load %alloc[%c8] : memref<13xindex>
  %c9 = arith.constant 9 : index
  %11 = memref.load %alloc[%c9] : memref<13xindex>
  %c10 = arith.constant 10 : index
  %12 = memref.load %alloc[%c10] : memref<13xindex>
  %c11 = arith.constant 11 : index
  %13 = memref.load %alloc[%c11] : memref<13xindex>
  %c12 = arith.constant 12 : index
  %14 = memref.load %alloc[%c12] : memref<13xindex>
  %alloc_8 = memref.alloc(%2) : memref<?xindex>
  %c0_9 = arith.constant 0 : index
  %c0_10 = arith.constant 0 : index
  %c1_11 = arith.constant 1 : index
  scf.for %arg0 = %c0_10 to %2 step %c1_11 {
    memref.store %c0_9, %alloc_8[%arg0] : memref<?xindex>
  }
  %cast_12 = memref.cast %alloc_8 : memref<?xindex> to memref<*xindex>
  %alloc_13 = memref.alloc(%3) : memref<?xindex>
  %c0_14 = arith.constant 0 : index
  %c0_15 = arith.constant 0 : index
  %c1_16 = arith.constant 1 : index
  scf.for %arg0 = %c0_15 to %3 step %c1_16 {
    memref.store %c0_14, %alloc_13[%arg0] : memref<?xindex>
  }
  %cast_17 = memref.cast %alloc_13 : memref<?xindex> to memref<*xindex>
  %alloc_18 = memref.alloc(%4) : memref<?xindex>
  %c0_19 = arith.constant 0 : index
  %c0_20 = arith.constant 0 : index
  %c1_21 = arith.constant 1 : index
  scf.for %arg0 = %c0_20 to %4 step %c1_21 {
    memref.store %c0_19, %alloc_18[%arg0] : memref<?xindex>
  }
  %cast_22 = memref.cast %alloc_18 : memref<?xindex> to memref<*xindex>
  %alloc_23 = memref.alloc(%5) : memref<?xindex>
  %c0_24 = arith.constant 0 : index
  %c0_25 = arith.constant 0 : index
  %c1_26 = arith.constant 1 : index
  scf.for %arg0 = %c0_25 to %5 step %c1_26 {
    memref.store %c0_24, %alloc_23[%arg0] : memref<?xindex>
  }
  %cast_27 = memref.cast %alloc_23 : memref<?xindex> to memref<*xindex>
  %alloc_28 = memref.alloc(%6) : memref<?xindex>
  %c0_29 = arith.constant 0 : index
  %c0_30 = arith.constant 0 : index
  %c1_31 = arith.constant 1 : index
  scf.for %arg0 = %c0_30 to %6 step %c1_31 {
    memref.store %c0_29, %alloc_28[%arg0] : memref<?xindex>
  }
  %cast_32 = memref.cast %alloc_28 : memref<?xindex> to memref<*xindex>
  %alloc_33 = memref.alloc(%7) : memref<?xindex>
  %c0_34 = arith.constant 0 : index
  %c0_35 = arith.constant 0 : index
  %c1_36 = arith.constant 1 : index
  scf.for %arg0 = %c0_35 to %7 step %c1_36 {
    memref.store %c0_34, %alloc_33[%arg0] : memref<?xindex>
  }
  %cast_37 = memref.cast %alloc_33 : memref<?xindex> to memref<*xindex>
  %alloc_38 = memref.alloc(%8) : memref<?xindex>
  %c0_39 = arith.constant 0 : index
  %c0_40 = arith.constant 0 : index
  %c1_41 = arith.constant 1 : index
  scf.for %arg0 = %c0_40 to %8 step %c1_41 {
    memref.store %c0_39, %alloc_38[%arg0] : memref<?xindex>
  }
  %cast_42 = memref.cast %alloc_38 : memref<?xindex> to memref<*xindex>
  %alloc_43 = memref.alloc(%9) : memref<?xindex>
  %c0_44 = arith.constant 0 : index
  %c0_45 = arith.constant 0 : index
  %c1_46 = arith.constant 1 : index
  scf.for %arg0 = %c0_45 to %9 step %c1_46 {
    memref.store %c0_44, %alloc_43[%arg0] : memref<?xindex>
  }
  %cast_47 = memref.cast %alloc_43 : memref<?xindex> to memref<*xindex>
  %alloc_48 = memref.alloc(%10) : memref<?xf64>
  %cst = arith.constant 0.000000e+00 : f64
  %c0_49 = arith.constant 0 : index
  %c1_50 = arith.constant 1 : index
  scf.for %arg0 = %c0_49 to %10 step %c1_50 {
    memref.store %cst, %alloc_48[%arg0] : memref<?xf64>
  }
  %cast_51 = memref.cast %alloc_48 : memref<?xf64> to memref<*xf64>
  call @read_input_2D_f64(%c0_i32, %c0_2, %c-1, %c1_3, %c-1, %cast_12, %cast_17, %cast_22, %cast_27, %cast_32, %cast_37, %cast_42, %cast_47, %cast_51, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
  %15 = bufferization.to_tensor %alloc_8 : memref<?xindex>
  %16 = bufferization.to_tensor %alloc_13 : memref<?xindex>
  %17 = bufferization.to_tensor %alloc_18 : memref<?xindex>
  %18 = bufferization.to_tensor %alloc_23 : memref<?xindex>
  %19 = bufferization.to_tensor %alloc_28 : memref<?xindex>
  %20 = bufferization.to_tensor %alloc_33 : memref<?xindex>
  %21 = bufferization.to_tensor %alloc_38 : memref<?xindex>
  %22 = bufferization.to_tensor %alloc_43 : memref<?xindex>
  %23 = bufferization.to_tensor %alloc_48 : memref<?xf64>
  %24 = ta.sptensor_construct(%15, %16, %17, %18, %19, %20, %21, %22, %23, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
  %25 = "ta.static_index_label"(%c0, %11, %c1) : (index, index, index) -> !ta.range
  %26 = "ta.static_index_label"(%c0_0, %12, %c1_1) : (index, index, index) -> !ta.range
  %27 = "ta.dense_tensor_decl"(%26, %25) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %28 = "ta.dense_tensor_decl"(%26, %25) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.fill"(%27) {value = 1.000000e+00 : f64} : (tensor<?x?xf64>) -> ()
  %29 = "it.ComputeRHS"(%24, %27) {allBlocks = [["UNK", "UNK"], ["UNK", "UNK"]], allFormats = [["D", "CU"], ["D", "D"]], allPerms = [[0, 1], [1, 0]]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>, tensor<?x?xf64>) -> tensor<*xf64>
  %30 = "it.ComputeLHS"(%28) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[1, 0]]} : (tensor<?x?xf64>) -> tensor<*xf64>
  %31 = "it.Compute"(%29, %30) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
  %32 = "it.Indices"(%31) {indices = [1]} : (i64) -> i64
  %33 = "it.Indices"(%32) {indices = [0]} : (i64) -> i64
  %34 = "it.itree"(%33) : (i64) -> i64
  "ta.print"(%24) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
  "ta.print"(%27) : (tensor<?x?xf64>) -> ()
  "ta.print"(%28) : (tensor<?x?xf64>) -> ()
  return
}

/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1056 --------------DenseTensorDeclarationLowering in format begin
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1058 

/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1060 %27 = "ta.dense_tensor_decl"(%26, %25) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
-----------------------------------
memref<?x?xf64>
-----------------------------------
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1077 %26 = "ta.static_index_label"(%c0_0, %12, %c1_1) : (index, index, index) -> !ta.range
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1083 %12 = memref.load %alloc[%c10] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1077 %25 = "ta.static_index_label"(%c0, %11, %c1) : (index, index, index) -> !ta.range
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1083 %11 = memref.load %alloc[%c9] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1092 %29 = "it.ComputeRHS"(%24, %27) {allBlocks = [["UNK", "UNK"], ["UNK", "UNK"]], allFormats = [["D", "CU"], ["D", "D"]], allPerms = [[0, 1], [1, 0]]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>, tensor<?x?xf64>) -> tensor<*xf64>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1092 "ta.print"(%27) : (tensor<?x?xf64>) -> ()
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1092 "ta.fill"(%27) {value = 1.000000e+00 : f64} : (tensor<?x?xf64>) -> ()
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1097  AllocOp for initialization is_filled: 1 
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1103 %alloc_52 = memref.alloc(%12, %11) : memref<?x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1115  TensorLoad:
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1116 %27 = bufferization.to_tensor %alloc_52 : memref<?x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1122 --------------DenseTensorDeclarationLowering in format end
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1123 

/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1056 --------------DenseTensorDeclarationLowering in format begin
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1058 

/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1060 %29 = "ta.dense_tensor_decl"(%26, %25) {format = "Dense"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
-----------------------------------
memref<?x?xf64>
-----------------------------------
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1077 %26 = "ta.static_index_label"(%c0_0, %12, %c1_1) : (index, index, index) -> !ta.range
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1083 %12 = memref.load %alloc[%c10] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1077 %25 = "ta.static_index_label"(%c0, %11, %c1) : (index, index, index) -> !ta.range
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1083 %11 = memref.load %alloc[%c9] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1092 %31 = "it.ComputeLHS"(%29) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[1, 0]]} : (tensor<?x?xf64>) -> tensor<*xf64>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1092 "ta.print"(%29) : (tensor<?x?xf64>) -> ()
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1097  AllocOp for initialization is_filled: 0 
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1109 %101 = "memref.alloc"(%36, %34) {operand_segment_sizes = array<i32: 2, 0>} : (index, index) -> memref<?x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1115  TensorLoad:
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1116 %105 = "bufferization.to_tensor"(%101) : (memref<?x?xf64>) -> tensor<?x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1122 --------------DenseTensorDeclarationLowering in format end
/home/patrick/Work/PNNL/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1123 

loc("first.ta":8:2): error: 'memref.store' op store index operand count not equal to memref rank
// -----// IR Dump After {anonymous}::DenseTensorDeclLoweringPass Failed () //----- //
"func.func"() ({
  %0 = "arith.constant"() {value = 0 : index} : () -> index
  %1 = "arith.constant"() {value = 1 : index} : () -> index
  %2 = "ta.dynamic_index_label"(%0, %1) : (index, index) -> !ta.range
  %3 = "arith.constant"() {value = 0 : index} : () -> index
  %4 = "arith.constant"() {value = 1 : index} : () -> index
  %5 = "ta.dynamic_index_label"(%3, %4) : (index, index) -> !ta.range
  %6 = "memref.alloc"() {operand_segment_sizes = array<i32: 0, 0>} : () -> memref<13xindex>
  %7 = "memref.cast"(%6) : (memref<13xindex>) -> memref<*xindex>
  %8 = "arith.constant"() {value = -1 : index} : () -> index
  %9 = "arith.constant"() {value = 0 : index} : () -> index
  %10 = "arith.constant"() {value = 1 : index} : () -> index
  %11 = "arith.constant"() {value = 2 : index} : () -> index
  %12 = "arith.constant"() {value = 3 : index} : () -> index
  %13 = "arith.constant"() {value = 0 : i32} : () -> i32
  %14 = "arith.constant"() {value = 1 : i32} : () -> i32
  "func.call"(%13, %9, %8, %10, %8, %7, %14) {callee = @read_input_sizes_2D_f64, filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
  %15 = "arith.constant"() {value = 0 : index} : () -> index
  %16 = "memref.load"(%6, %15) {nontemporal = false} : (memref<13xindex>, index) -> index
  %17 = "arith.constant"() {value = 1 : index} : () -> index
  %18 = "memref.load"(%6, %17) {nontemporal = false} : (memref<13xindex>, index) -> index
  %19 = "arith.constant"() {value = 2 : index} : () -> index
  %20 = "memref.load"(%6, %19) {nontemporal = false} : (memref<13xindex>, index) -> index
  %21 = "arith.constant"() {value = 3 : index} : () -> index
  %22 = "memref.load"(%6, %21) {nontemporal = false} : (memref<13xindex>, index) -> index
  %23 = "arith.constant"() {value = 4 : index} : () -> index
  %24 = "memref.load"(%6, %23) {nontemporal = false} : (memref<13xindex>, index) -> index
  %25 = "arith.constant"() {value = 5 : index} : () -> index
  %26 = "memref.load"(%6, %25) {nontemporal = false} : (memref<13xindex>, index) -> index
  %27 = "arith.constant"() {value = 6 : index} : () -> index
  %28 = "memref.load"(%6, %27) {nontemporal = false} : (memref<13xindex>, index) -> index
  %29 = "arith.constant"() {value = 7 : index} : () -> index
  %30 = "memref.load"(%6, %29) {nontemporal = false} : (memref<13xindex>, index) -> index
  %31 = "arith.constant"() {value = 8 : index} : () -> index
  %32 = "memref.load"(%6, %31) {nontemporal = false} : (memref<13xindex>, index) -> index
  %33 = "arith.constant"() {value = 9 : index} : () -> index
  %34 = "memref.load"(%6, %33) {nontemporal = false} : (memref<13xindex>, index) -> index
  %35 = "arith.constant"() {value = 10 : index} : () -> index
  %36 = "memref.load"(%6, %35) {nontemporal = false} : (memref<13xindex>, index) -> index
  %37 = "arith.constant"() {value = 11 : index} : () -> index
  %38 = "memref.load"(%6, %37) {nontemporal = false} : (memref<13xindex>, index) -> index
  %39 = "arith.constant"() {value = 12 : index} : () -> index
  %40 = "memref.load"(%6, %39) {nontemporal = false} : (memref<13xindex>, index) -> index
  %41 = "memref.alloc"(%16) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %42 = "arith.constant"() {value = 0 : index} : () -> index
  %43 = "arith.constant"() {value = 0 : index} : () -> index
  %44 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%43, %16, %44) ({
  ^bb0(%arg0: index):
    "memref.store"(%42, %41, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %45 = "memref.cast"(%41) : (memref<?xindex>) -> memref<*xindex>
  %46 = "memref.alloc"(%18) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %47 = "arith.constant"() {value = 0 : index} : () -> index
  %48 = "arith.constant"() {value = 0 : index} : () -> index
  %49 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%48, %18, %49) ({
  ^bb0(%arg0: index):
    "memref.store"(%47, %46, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %50 = "memref.cast"(%46) : (memref<?xindex>) -> memref<*xindex>
  %51 = "memref.alloc"(%20) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %52 = "arith.constant"() {value = 0 : index} : () -> index
  %53 = "arith.constant"() {value = 0 : index} : () -> index
  %54 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%53, %20, %54) ({
  ^bb0(%arg0: index):
    "memref.store"(%52, %51, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %55 = "memref.cast"(%51) : (memref<?xindex>) -> memref<*xindex>
  %56 = "memref.alloc"(%22) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %57 = "arith.constant"() {value = 0 : index} : () -> index
  %58 = "arith.constant"() {value = 0 : index} : () -> index
  %59 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%58, %22, %59) ({
  ^bb0(%arg0: index):
    "memref.store"(%57, %56, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %60 = "memref.cast"(%56) : (memref<?xindex>) -> memref<*xindex>
  %61 = "memref.alloc"(%24) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %62 = "arith.constant"() {value = 0 : index} : () -> index
  %63 = "arith.constant"() {value = 0 : index} : () -> index
  %64 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%63, %24, %64) ({
  ^bb0(%arg0: index):
    "memref.store"(%62, %61, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %65 = "memref.cast"(%61) : (memref<?xindex>) -> memref<*xindex>
  %66 = "memref.alloc"(%26) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %67 = "arith.constant"() {value = 0 : index} : () -> index
  %68 = "arith.constant"() {value = 0 : index} : () -> index
  %69 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%68, %26, %69) ({
  ^bb0(%arg0: index):
    "memref.store"(%67, %66, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %70 = "memref.cast"(%66) : (memref<?xindex>) -> memref<*xindex>
  %71 = "memref.alloc"(%28) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %72 = "arith.constant"() {value = 0 : index} : () -> index
  %73 = "arith.constant"() {value = 0 : index} : () -> index
  %74 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%73, %28, %74) ({
  ^bb0(%arg0: index):
    "memref.store"(%72, %71, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %75 = "memref.cast"(%71) : (memref<?xindex>) -> memref<*xindex>
  %76 = "memref.alloc"(%30) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %77 = "arith.constant"() {value = 0 : index} : () -> index
  %78 = "arith.constant"() {value = 0 : index} : () -> index
  %79 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%78, %30, %79) ({
  ^bb0(%arg0: index):
    "memref.store"(%77, %76, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %80 = "memref.cast"(%76) : (memref<?xindex>) -> memref<*xindex>
  %81 = "memref.alloc"(%32) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xf64>
  %82 = "arith.constant"() {value = 0.000000e+00 : f64} : () -> f64
  %83 = "arith.constant"() {value = 0 : index} : () -> index
  %84 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%83, %32, %84) ({
  ^bb0(%arg0: index):
    "memref.store"(%82, %81, %arg0) {nontemporal = false} : (f64, memref<?xf64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %85 = "memref.cast"(%81) : (memref<?xf64>) -> memref<*xf64>
  "func.call"(%13, %9, %8, %10, %8, %45, %50, %55, %60, %65, %70, %75, %80, %85, %14) {callee = @read_input_2D_f64, filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
  %86 = "bufferization.to_tensor"(%41) : (memref<?xindex>) -> tensor<?xindex>
  %87 = "bufferization.to_tensor"(%46) : (memref<?xindex>) -> tensor<?xindex>
  %88 = "bufferization.to_tensor"(%51) : (memref<?xindex>) -> tensor<?xindex>
  %89 = "bufferization.to_tensor"(%56) : (memref<?xindex>) -> tensor<?xindex>
  %90 = "bufferization.to_tensor"(%61) : (memref<?xindex>) -> tensor<?xindex>
  %91 = "bufferization.to_tensor"(%66) : (memref<?xindex>) -> tensor<?xindex>
  %92 = "bufferization.to_tensor"(%71) : (memref<?xindex>) -> tensor<?xindex>
  %93 = "bufferization.to_tensor"(%76) : (memref<?xindex>) -> tensor<?xindex>
  %94 = "bufferization.to_tensor"(%81) : (memref<?xf64>) -> tensor<?xf64>
  %95 = "ta.sptensor_construct"(%86, %87, %88, %89, %90, %91, %92, %93, %94, %16, %18, %20, %22, %24, %26, %28, %30, %32, %34, %36) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>
  %96 = "ta.static_index_label"(%0, %34, %1) : (index, index, index) -> !ta.range
  %97 = "ta.static_index_label"(%3, %36, %4) : (index, index, index) -> !ta.range
  %98 = "memref.alloc"(%36, %34) {alignment = 32 : i64, operand_segment_sizes = array<i32: 2, 0>} : (index, index) -> memref<?x?xf64>
  %99 = "bufferization.to_tensor"(%98) : (memref<?x?xf64>) -> tensor<?x?xf64>
  %100 = "memref.alloc"(%36, %34) {alignment = 32 : i64, operand_segment_sizes = array<i32: 2, 0>} : (index, index) -> memref<?x?xf64>
  %101 = "arith.constant"() {value = 0.000000e+00 : f64} : () -> f64
  %102 = "arith.constant"() {value = 0 : index} : () -> index
  %103 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%102, %36, %103) ({
  ^bb0(%arg0: index):
    "memref.store"(%101, %100, %arg0) {nontemporal = false} : (f64, memref<?x?xf64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %104 = "bufferization.to_tensor"(%100) : (memref<?x?xf64>) -> tensor<?x?xf64>
  "ta.fill"(%99) {value = 1.000000e+00 : f64} : (tensor<?x?xf64>) -> ()
  %105 = "it.ComputeRHS"(%95, %99) {allBlocks = [["UNK", "UNK"], ["UNK", "UNK"]], allFormats = [["D", "CU"], ["D", "D"]], allPerms = [[0, 1], [1, 0]]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>, tensor<?x?xf64>) -> tensor<*xf64>
  %106 = "it.ComputeLHS"(%104) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[1, 0]]} : (tensor<?x?xf64>) -> tensor<*xf64>
  %107 = "it.Compute"(%105, %106) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
  %108 = "it.Indices"(%107) {indices = [1]} : (i64) -> i64
  %109 = "it.Indices"(%108) {indices = [0]} : (i64) -> i64
  %110 = "it.itree"(%109) : (i64) -> i64
  "ta.print"(%95) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
  "ta.print"(%99) : (tensor<?x?xf64>) -> ()
  "ta.print"(%104) : (tensor<?x?xf64>) -> ()
  "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "main"} : () -> ()

