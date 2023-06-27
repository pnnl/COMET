/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1058  SparseInputTensorDeclOpLowering in format begin
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1059 %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "ELL", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1070  --- ELL
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1072  2
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1082 
Check the tensor is input or output
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1083 "ta.print"(%2) : (tensor<?x?xf64>) -> ()
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1175  the tensor is in PrintOp
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1082 
Check the tensor is input or output
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1083 "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME1", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1171  the tensor is in fill_from_file op
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1202  isOutputTensor: 0
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1211  ELL isDense: 0
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1216  Sparse input tensor 
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1272  filename: SPARSE_FILE_NAME1
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1276  SPARSE_FILE_NAME1
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1279  readMode: 1
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1285  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1286 %alloc = memref.alloc() : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1291  Get the dim_format
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1310  Parsed fileID: 1
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1329  2D
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:299 Inserting insertReadFileLibCall
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:319  Rank Size is 2
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:350 Adding read_input_2D_f64 to the module
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:377 Adding read_input_sizes_2D_f64 to the module
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1382  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1383 %2 = memref.load %alloc[%c0_5] : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1382  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1383 %3 = memref.load %alloc[%c1_6] : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1382  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1383 %4 = memref.load %alloc[%c2_7] : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1382  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1383 %5 = memref.load %alloc[%c3_8] : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1382  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1383 %6 = memref.load %alloc[%c4] : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1382  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1383 %7 = memref.load %alloc[%c5] : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1382  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1383 %8 = memref.load %alloc[%c6] : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1396 %2 = memref.load %alloc[%c0_5] : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1398  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1399 %alloc_9 = memref.alloc(%2) : memref<?xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1396 %3 = memref.load %alloc[%c1_6] : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1398  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1399 %alloc_14 = memref.alloc(%3) : memref<?xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1396 %4 = memref.load %alloc[%c2_7] : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1398  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1399 %alloc_19 = memref.alloc(%4) : memref<?xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1396 %5 = memref.load %alloc[%c3_8] : memref<7xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1398  /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1399 %alloc_24 = memref.alloc(%5) : memref<?xindex>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1413 > /Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1414 %alloc_29 = memref.alloc(%6) : memref<?xf64>
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1474  Generate read_input_2D or read_input_3D functions
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1513 SparseTensorConstructOp generated for input sparse tensor:
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1514 %14 = ta.sptensor_construct(%9, %10, %11, %12, %13, %2, %3, %4, %5, %6, %7, %8) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index>)
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1522 %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1523 %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1532 %15 = "ta.static_index_label"(%c0, %7, %c1) : (index, index, index) -> !ta.range
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1522 %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1523 %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1532 %16 = "ta.static_index_label"(%c0_0, %8, %c1_1) : (index, index, index) -> !ta.range
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1578  SparseInputTensorDeclOpLowering in format end
/Users/flyn908/Projects/COMET/lib/Dialect/TensorAlgebra/Transforms/TensorDeclLowering.cpp:1720 ---------------SparseTensorDeclLoweringPass end
module {
  func.func @main() {
    %alloc = memref.alloc() : memref<7xindex>
    %cast = memref.cast %alloc : memref<7xindex> to memref<*xindex>
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1_i32 = arith.constant 1 : i32
    call @read_input_sizes_2D_f64(%c1_i32, %c0, %c0, %c3, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME1"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
    %0 = memref.load %alloc[%c0] : memref<7xindex>
    %c1 = arith.constant 1 : index
    %1 = memref.load %alloc[%c1] : memref<7xindex>
    %c2 = arith.constant 2 : index
    %2 = memref.load %alloc[%c2] : memref<7xindex>
    %3 = memref.load %alloc[%c3] : memref<7xindex>
    %c4 = arith.constant 4 : index
    %4 = memref.load %alloc[%c4] : memref<7xindex>
    %c5 = arith.constant 5 : index
    %5 = memref.load %alloc[%c5] : memref<7xindex>
    %c6 = arith.constant 6 : index
    %6 = memref.load %alloc[%c6] : memref<7xindex>
    %alloc_0 = memref.alloc(%0) : memref<?xindex>
    scf.for %arg0 = %c0 to %0 step %c1 {
      memref.store %c0, %alloc_0[%arg0] : memref<?xindex>
    }
    %cast_1 = memref.cast %alloc_0 : memref<?xindex> to memref<*xindex>
    %alloc_2 = memref.alloc(%1) : memref<?xindex>
    scf.for %arg0 = %c0 to %1 step %c1 {
      memref.store %c0, %alloc_2[%arg0] : memref<?xindex>
    }
    %cast_3 = memref.cast %alloc_2 : memref<?xindex> to memref<*xindex>
    %alloc_4 = memref.alloc(%2) : memref<?xindex>
    scf.for %arg0 = %c0 to %2 step %c1 {
      memref.store %c0, %alloc_4[%arg0] : memref<?xindex>
    }
    %cast_5 = memref.cast %alloc_4 : memref<?xindex> to memref<*xindex>
    %alloc_6 = memref.alloc(%3) : memref<?xindex>
    scf.for %arg0 = %c0 to %3 step %c1 {
      memref.store %c0, %alloc_6[%arg0] : memref<?xindex>
    }
    %cast_7 = memref.cast %alloc_6 : memref<?xindex> to memref<*xindex>
    %alloc_8 = memref.alloc(%4) : memref<?xf64>
    %cst = arith.constant 0.000000e+00 : f64
    scf.for %arg0 = %c0 to %4 step %c1 {
      memref.store %cst, %alloc_8[%arg0] : memref<?xf64>
    }
    %cast_9 = memref.cast %alloc_8 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c1_i32, %c0, %c0, %c3, %c-1, %cast_1, %cast_3, %cast_5, %cast_7, %cast_1, %cast_3, %cast_5, %cast_7, %cast_9, %c1_i32) {filename = "SPARSE_FILE_NAME1"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %7 = bufferization.to_tensor %alloc_0 : memref<?xindex>
    %8 = bufferization.to_tensor %alloc_2 : memref<?xindex>
    %9 = bufferization.to_tensor %alloc_4 : memref<?xindex>
    %10 = bufferization.to_tensor %alloc_6 : memref<?xindex>
    %11 = bufferization.to_tensor %alloc_8 : memref<?xf64>
    call @comet_print_memref_i64(%cast_1) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_3) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_5) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_7) : (memref<*xindex>) -> ()
    call @comet_print_memref_f64(%cast_9) : (memref<*xf64>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @comet_print_memref_f64(memref<*xf64>)
  func.func private @comet_print_memref_i64(memref<*xindex>)
}
IN0: 0
IN0: 1
IN0: 2
IN0: 3
IN0: 4
IN0: 5
IN0: 6
ArraySizes: 7
IN1: 0
IN1: 1
IN1: 2
IN1: 3
IN2: 4
