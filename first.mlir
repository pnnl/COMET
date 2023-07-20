// -----// IR Dump After (anonymous namespace)::FuncOpLoweringPass () //----- //
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
    %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
    %3 = "ta.sparse_tensor_decl"(%1, %0) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
    "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
    %4 = "ta.transpose"(%2, %1, %0) {formats = ["COO", "COO"], indexing_maps = [#map, #map1]} : (tensor<?x?xf64>, !ta.range, !ta.range) -> tensor<?x?xf64>
    "ta.set_op"(%4, %3) : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
    "ta.print"(%3) : (tensor<?x?xf64>) -> ()
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
  %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %3 = "ta.sparse_tensor_decl"(%1, %0) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
  %4 = "ta.transpose"(%2, %1, %0) {formats = ["COO", "COO"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]} : (tensor<?x?xf64>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%4, %3) : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
  "ta.print"(%3) : (tensor<?x?xf64>) -> ()
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
  %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %3 = "ta.sparse_tensor_decl"(%1, %0) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
  %4 = "ta.transpose"(%2, %1, %0) {formats = ["COO", "COO"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]} : (tensor<?x?xf64>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%4, %3) : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
  "ta.print"(%3) : (tensor<?x?xf64>) -> ()
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
  %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %3 = "ta.sparse_tensor_decl"(%1, %0) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
  %4 = "ta.transpose"(%2, %1, %0) {formats = ["COO", "COO"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]} : (tensor<?x?xf64>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%4, %3) : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
  "ta.print"(%3) : (tensor<?x?xf64>) -> ()
  return
}

// -----// IR Dump After (anonymous namespace)::LowerTensorAlgebraToIndexTreePass () //----- //
func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
  %2 = "ta.sparse_tensor_decl"(%0, %1) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %3 = "ta.sparse_tensor_decl"(%1, %0) {format = "COO", temporal_tensor = false} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.fill_from_file"(%2) {filename = "SPARSE_FILE_NAME0", readMode = 1 : i32} : (tensor<?x?xf64>) -> ()
  %4 = "ta.transpose"(%2, %1, %0) {formats = ["COO", "COO"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]} : (tensor<?x?xf64>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%4, %3) : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
  "ta.print"(%3) : (tensor<?x?xf64>) -> ()
  return
}

// -----// IR Dump After (anonymous namespace)::SparseTensorDeclLoweringPass () //----- //
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
  call @read_input_sizes_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
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
  call @read_input_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast_12, %cast_17, %cast_22, %cast_27, %cast_32, %cast_37, %cast_42, %cast_47, %cast_51, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
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
  %27 = "ta.sparse_output_tensor_decl"(%26, %25) {format = "COO"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %28 = "ta.transpose"(%24, %26, %25) {formats = ["COO", "COO"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%28, %27) : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
  "ta.print"(%27) : (tensor<?x?xf64>) -> ()
  return
}

// -----// IR Dump After (anonymous namespace)::DenseTensorDeclLoweringPass () //----- //
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
  call @read_input_sizes_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
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
  call @read_input_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast_12, %cast_17, %cast_22, %cast_27, %cast_32, %cast_37, %cast_42, %cast_47, %cast_51, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
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
  %27 = "ta.sparse_output_tensor_decl"(%26, %25) {format = "COO"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %28 = "ta.transpose"(%24, %26, %25) {formats = ["COO", "COO"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%28, %27) : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
  "ta.print"(%27) : (tensor<?x?xf64>) -> ()
  return
}

// -----// IR Dump After (anonymous namespace)::TensorFillLoweringPass () //----- //
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
  call @read_input_sizes_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
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
  call @read_input_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast_12, %cast_17, %cast_22, %cast_27, %cast_32, %cast_37, %cast_42, %cast_47, %cast_51, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
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
  %27 = "ta.sparse_output_tensor_decl"(%26, %25) {format = "COO"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %28 = "ta.transpose"(%24, %26, %25) {formats = ["COO", "COO"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%28, %27) : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
  "ta.print"(%27) : (tensor<?x?xf64>) -> ()
  return
}

// -----// IR Dump After (anonymous namespace)::DenseTensorDeclLoweringPass () //----- //
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
  call @read_input_sizes_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
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
  call @read_input_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast_12, %cast_17, %cast_22, %cast_27, %cast_32, %cast_37, %cast_42, %cast_47, %cast_51, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
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
  %27 = "ta.sparse_output_tensor_decl"(%26, %25) {format = "COO"} : (!ta.range, !ta.range) -> tensor<?x?xf64>
  %28 = "ta.transpose"(%24, %26, %25) {formats = ["COO", "COO"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%28, %27) : (tensor<?x?xf64>, tensor<?x?xf64>) -> ()
  "ta.print"(%27) : (tensor<?x?xf64>) -> ()
  return
}

// -----// IR Dump After (anonymous namespace)::SparseOutputTensorDeclLoweringPass () //----- //
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
  call @read_input_sizes_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
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
  call @read_input_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast_12, %cast_17, %cast_22, %cast_27, %cast_32, %cast_37, %cast_42, %cast_47, %cast_51, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
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
  %c1_52 = arith.constant 1 : index
  %c2_53 = arith.constant 2 : index
  %alloc_54 = memref.alloc(%c2_53) : memref<?xindex>
  %c0_55 = arith.constant 0 : index
  %c0_56 = arith.constant 0 : index
  %c1_57 = arith.constant 1 : index
  scf.for %arg0 = %c0_56 to %c2_53 step %c1_57 {
    memref.store %c0_55, %alloc_54[%arg0] : memref<?xindex>
  }
  %27 = bufferization.to_tensor %alloc_54 : memref<?xindex>
  %alloc_58 = memref.alloc(%7) : memref<?xindex>
  %c0_59 = arith.constant 0 : index
  %c0_60 = arith.constant 0 : index
  %c1_61 = arith.constant 1 : index
  scf.for %arg0 = %c0_60 to %7 step %c1_61 {
    memref.store %c0_59, %alloc_58[%arg0] : memref<?xindex>
  }
  %28 = bufferization.to_tensor %alloc_58 : memref<?xindex>
  %alloc_62 = memref.alloc(%c1_52) : memref<?xindex>
  %c0_63 = arith.constant 0 : index
  %c0_64 = arith.constant 0 : index
  %c1_65 = arith.constant 1 : index
  scf.for %arg0 = %c0_64 to %c1_52 step %c1_65 {
    memref.store %c0_63, %alloc_62[%arg0] : memref<?xindex>
  }
  %29 = bufferization.to_tensor %alloc_62 : memref<?xindex>
  %alloc_66 = memref.alloc(%9) : memref<?xindex>
  %c0_67 = arith.constant 0 : index
  %c0_68 = arith.constant 0 : index
  %c1_69 = arith.constant 1 : index
  scf.for %arg0 = %c0_68 to %9 step %c1_69 {
    memref.store %c0_67, %alloc_66[%arg0] : memref<?xindex>
  }
  %30 = bufferization.to_tensor %alloc_66 : memref<?xindex>
  %alloc_70 = memref.alloc(%c1_52) : memref<?xindex>
  %c0_71 = arith.constant 0 : index
  %c0_72 = arith.constant 0 : index
  %c1_73 = arith.constant 1 : index
  scf.for %arg0 = %c0_72 to %c1_52 step %c1_73 {
    memref.store %c0_71, %alloc_70[%arg0] : memref<?xindex>
  }
  %31 = bufferization.to_tensor %alloc_70 : memref<?xindex>
  %alloc_74 = memref.alloc(%3) : memref<?xindex>
  %c0_75 = arith.constant 0 : index
  %c0_76 = arith.constant 0 : index
  %c1_77 = arith.constant 1 : index
  scf.for %arg0 = %c0_76 to %3 step %c1_77 {
    memref.store %c0_75, %alloc_74[%arg0] : memref<?xindex>
  }
  %32 = bufferization.to_tensor %alloc_74 : memref<?xindex>
  %alloc_78 = memref.alloc(%c1_52) : memref<?xindex>
  %c0_79 = arith.constant 0 : index
  %c0_80 = arith.constant 0 : index
  %c1_81 = arith.constant 1 : index
  scf.for %arg0 = %c0_80 to %c1_52 step %c1_81 {
    memref.store %c0_79, %alloc_78[%arg0] : memref<?xindex>
  }
  %33 = bufferization.to_tensor %alloc_78 : memref<?xindex>
  %alloc_82 = memref.alloc(%5) : memref<?xindex>
  %c0_83 = arith.constant 0 : index
  %c0_84 = arith.constant 0 : index
  %c1_85 = arith.constant 1 : index
  scf.for %arg0 = %c0_84 to %5 step %c1_85 {
    memref.store %c0_83, %alloc_82[%arg0] : memref<?xindex>
  }
  %34 = bufferization.to_tensor %alloc_82 : memref<?xindex>
  %alloc_86 = memref.alloc(%10) : memref<?xf64>
  %cst_87 = arith.constant 0.000000e+00 : f64
  %c0_88 = arith.constant 0 : index
  %c1_89 = arith.constant 1 : index
  scf.for %arg0 = %c0_88 to %10 step %c1_89 {
    memref.store %cst_87, %alloc_86[%arg0] : memref<?xf64>
  }
  %35 = bufferization.to_tensor %alloc_86 : memref<?xf64>
  %36 = ta.sptensor_construct(%27, %28, %29, %30, %31, %32, %33, %34, %35, %c2_53, %7, %c1_52, %9, %c1_52, %3, %c1_52, %5, %10, %12, %11) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
  %37 = "ta.transpose"(%24, %26, %25) {formats = ["COO", "COO"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%37, %36) : (tensor<?x?xf64>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
  "ta.print"(%36) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
  return
}

// -----// IR Dump After (anonymous namespace)::TensorFillLoweringPass () //----- //
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
  call @read_input_sizes_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
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
  call @read_input_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast_12, %cast_17, %cast_22, %cast_27, %cast_32, %cast_37, %cast_42, %cast_47, %cast_51, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
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
  %c1_52 = arith.constant 1 : index
  %c2_53 = arith.constant 2 : index
  %alloc_54 = memref.alloc(%c2_53) : memref<?xindex>
  %c0_55 = arith.constant 0 : index
  %c0_56 = arith.constant 0 : index
  %c1_57 = arith.constant 1 : index
  scf.for %arg0 = %c0_56 to %c2_53 step %c1_57 {
    memref.store %c0_55, %alloc_54[%arg0] : memref<?xindex>
  }
  %27 = bufferization.to_tensor %alloc_54 : memref<?xindex>
  %alloc_58 = memref.alloc(%7) : memref<?xindex>
  %c0_59 = arith.constant 0 : index
  %c0_60 = arith.constant 0 : index
  %c1_61 = arith.constant 1 : index
  scf.for %arg0 = %c0_60 to %7 step %c1_61 {
    memref.store %c0_59, %alloc_58[%arg0] : memref<?xindex>
  }
  %28 = bufferization.to_tensor %alloc_58 : memref<?xindex>
  %alloc_62 = memref.alloc(%c1_52) : memref<?xindex>
  %c0_63 = arith.constant 0 : index
  %c0_64 = arith.constant 0 : index
  %c1_65 = arith.constant 1 : index
  scf.for %arg0 = %c0_64 to %c1_52 step %c1_65 {
    memref.store %c0_63, %alloc_62[%arg0] : memref<?xindex>
  }
  %29 = bufferization.to_tensor %alloc_62 : memref<?xindex>
  %alloc_66 = memref.alloc(%9) : memref<?xindex>
  %c0_67 = arith.constant 0 : index
  %c0_68 = arith.constant 0 : index
  %c1_69 = arith.constant 1 : index
  scf.for %arg0 = %c0_68 to %9 step %c1_69 {
    memref.store %c0_67, %alloc_66[%arg0] : memref<?xindex>
  }
  %30 = bufferization.to_tensor %alloc_66 : memref<?xindex>
  %alloc_70 = memref.alloc(%c1_52) : memref<?xindex>
  %c0_71 = arith.constant 0 : index
  %c0_72 = arith.constant 0 : index
  %c1_73 = arith.constant 1 : index
  scf.for %arg0 = %c0_72 to %c1_52 step %c1_73 {
    memref.store %c0_71, %alloc_70[%arg0] : memref<?xindex>
  }
  %31 = bufferization.to_tensor %alloc_70 : memref<?xindex>
  %alloc_74 = memref.alloc(%3) : memref<?xindex>
  %c0_75 = arith.constant 0 : index
  %c0_76 = arith.constant 0 : index
  %c1_77 = arith.constant 1 : index
  scf.for %arg0 = %c0_76 to %3 step %c1_77 {
    memref.store %c0_75, %alloc_74[%arg0] : memref<?xindex>
  }
  %32 = bufferization.to_tensor %alloc_74 : memref<?xindex>
  %alloc_78 = memref.alloc(%c1_52) : memref<?xindex>
  %c0_79 = arith.constant 0 : index
  %c0_80 = arith.constant 0 : index
  %c1_81 = arith.constant 1 : index
  scf.for %arg0 = %c0_80 to %c1_52 step %c1_81 {
    memref.store %c0_79, %alloc_78[%arg0] : memref<?xindex>
  }
  %33 = bufferization.to_tensor %alloc_78 : memref<?xindex>
  %alloc_82 = memref.alloc(%5) : memref<?xindex>
  %c0_83 = arith.constant 0 : index
  %c0_84 = arith.constant 0 : index
  %c1_85 = arith.constant 1 : index
  scf.for %arg0 = %c0_84 to %5 step %c1_85 {
    memref.store %c0_83, %alloc_82[%arg0] : memref<?xindex>
  }
  %34 = bufferization.to_tensor %alloc_82 : memref<?xindex>
  %alloc_86 = memref.alloc(%10) : memref<?xf64>
  %cst_87 = arith.constant 0.000000e+00 : f64
  %c0_88 = arith.constant 0 : index
  %c1_89 = arith.constant 1 : index
  scf.for %arg0 = %c0_88 to %10 step %c1_89 {
    memref.store %cst_87, %alloc_86[%arg0] : memref<?xf64>
  }
  %35 = bufferization.to_tensor %alloc_86 : memref<?xf64>
  %36 = ta.sptensor_construct(%27, %28, %29, %30, %31, %32, %33, %34, %35, %c2_53, %7, %c1_52, %9, %c1_52, %3, %c1_52, %5, %10, %12, %11) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
  %37 = "ta.transpose"(%24, %26, %25) {formats = ["COO", "COO"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%37, %36) : (tensor<?x?xf64>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
  "ta.print"(%36) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
  return
}

// -----// IR Dump After (anonymous namespace)::PCToLoopsLoweringPass () //----- //
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
  call @read_input_sizes_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
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
  call @read_input_2D_f64(%c0_i32, %c2, %c-1, %c3, %c-1, %cast_12, %cast_17, %cast_22, %cast_27, %cast_32, %cast_37, %cast_42, %cast_47, %cast_51, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
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
  %c1_52 = arith.constant 1 : index
  %c2_53 = arith.constant 2 : index
  %alloc_54 = memref.alloc(%c2_53) : memref<?xindex>
  %c0_55 = arith.constant 0 : index
  %c0_56 = arith.constant 0 : index
  %c1_57 = arith.constant 1 : index
  scf.for %arg0 = %c0_56 to %c2_53 step %c1_57 {
    memref.store %c0_55, %alloc_54[%arg0] : memref<?xindex>
  }
  %27 = bufferization.to_tensor %alloc_54 : memref<?xindex>
  %alloc_58 = memref.alloc(%7) : memref<?xindex>
  %c0_59 = arith.constant 0 : index
  %c0_60 = arith.constant 0 : index
  %c1_61 = arith.constant 1 : index
  scf.for %arg0 = %c0_60 to %7 step %c1_61 {
    memref.store %c0_59, %alloc_58[%arg0] : memref<?xindex>
  }
  %28 = bufferization.to_tensor %alloc_58 : memref<?xindex>
  %alloc_62 = memref.alloc(%c1_52) : memref<?xindex>
  %c0_63 = arith.constant 0 : index
  %c0_64 = arith.constant 0 : index
  %c1_65 = arith.constant 1 : index
  scf.for %arg0 = %c0_64 to %c1_52 step %c1_65 {
    memref.store %c0_63, %alloc_62[%arg0] : memref<?xindex>
  }
  %29 = bufferization.to_tensor %alloc_62 : memref<?xindex>
  %alloc_66 = memref.alloc(%9) : memref<?xindex>
  %c0_67 = arith.constant 0 : index
  %c0_68 = arith.constant 0 : index
  %c1_69 = arith.constant 1 : index
  scf.for %arg0 = %c0_68 to %9 step %c1_69 {
    memref.store %c0_67, %alloc_66[%arg0] : memref<?xindex>
  }
  %30 = bufferization.to_tensor %alloc_66 : memref<?xindex>
  %alloc_70 = memref.alloc(%c1_52) : memref<?xindex>
  %c0_71 = arith.constant 0 : index
  %c0_72 = arith.constant 0 : index
  %c1_73 = arith.constant 1 : index
  scf.for %arg0 = %c0_72 to %c1_52 step %c1_73 {
    memref.store %c0_71, %alloc_70[%arg0] : memref<?xindex>
  }
  %31 = bufferization.to_tensor %alloc_70 : memref<?xindex>
  %alloc_74 = memref.alloc(%3) : memref<?xindex>
  %c0_75 = arith.constant 0 : index
  %c0_76 = arith.constant 0 : index
  %c1_77 = arith.constant 1 : index
  scf.for %arg0 = %c0_76 to %3 step %c1_77 {
    memref.store %c0_75, %alloc_74[%arg0] : memref<?xindex>
  }
  %32 = bufferization.to_tensor %alloc_74 : memref<?xindex>
  %alloc_78 = memref.alloc(%c1_52) : memref<?xindex>
  %c0_79 = arith.constant 0 : index
  %c0_80 = arith.constant 0 : index
  %c1_81 = arith.constant 1 : index
  scf.for %arg0 = %c0_80 to %c1_52 step %c1_81 {
    memref.store %c0_79, %alloc_78[%arg0] : memref<?xindex>
  }
  %33 = bufferization.to_tensor %alloc_78 : memref<?xindex>
  %alloc_82 = memref.alloc(%5) : memref<?xindex>
  %c0_83 = arith.constant 0 : index
  %c0_84 = arith.constant 0 : index
  %c1_85 = arith.constant 1 : index
  scf.for %arg0 = %c0_84 to %5 step %c1_85 {
    memref.store %c0_83, %alloc_82[%arg0] : memref<?xindex>
  }
  %34 = bufferization.to_tensor %alloc_82 : memref<?xindex>
  %alloc_86 = memref.alloc(%10) : memref<?xf64>
  %cst_87 = arith.constant 0.000000e+00 : f64
  %c0_88 = arith.constant 0 : index
  %c1_89 = arith.constant 1 : index
  scf.for %arg0 = %c0_88 to %10 step %c1_89 {
    memref.store %cst_87, %alloc_86[%arg0] : memref<?xf64>
  }
  %35 = bufferization.to_tensor %alloc_86 : memref<?xf64>
  %36 = ta.sptensor_construct(%27, %28, %29, %30, %31, %32, %33, %34, %35, %c2_53, %7, %c1_52, %9, %c1_52, %3, %c1_52, %5, %10, %12, %11) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
  %37 = "ta.transpose"(%24, %26, %25) {formats = ["COO", "COO"], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>]} : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>, !ta.range, !ta.range) -> tensor<?x?xf64>
  "ta.set_op"(%37, %36) : (tensor<?x?xf64>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
  "ta.print"(%36) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
  return
}

loc("first.ta":19:12): error: 'memref.cast' op operand type 'memref<?xindex>' and result type 'memref<*xf64>' are cast incompatible
// -----// IR Dump After (anonymous namespace)::LowerTensorAlgebraToSCFPass Failed () //----- //
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
  "func.call"(%13, %11, %8, %12, %8, %7, %14) {callee = @read_input_sizes_2D_f64, filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
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
  "func.call"(%13, %11, %8, %12, %8, %45, %50, %55, %60, %65, %70, %75, %80, %85, %14) {callee = @read_input_2D_f64, filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
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
  %98 = "arith.constant"() {value = 1 : index} : () -> index
  %99 = "arith.constant"() {value = 2 : index} : () -> index
  %100 = "memref.alloc"(%99) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %101 = "arith.constant"() {value = 0 : index} : () -> index
  %102 = "arith.constant"() {value = 0 : index} : () -> index
  %103 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%102, %99, %103) ({
  ^bb0(%arg0: index):
    "memref.store"(%101, %100, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %104 = "bufferization.to_tensor"(%100) : (memref<?xindex>) -> tensor<?xindex>
  %105 = "memref.alloc"(%26) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %106 = "arith.constant"() {value = 0 : index} : () -> index
  %107 = "arith.constant"() {value = 0 : index} : () -> index
  %108 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%107, %26, %108) ({
  ^bb0(%arg0: index):
    "memref.store"(%106, %105, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %109 = "bufferization.to_tensor"(%105) : (memref<?xindex>) -> tensor<?xindex>
  %110 = "memref.alloc"(%98) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %111 = "arith.constant"() {value = 0 : index} : () -> index
  %112 = "arith.constant"() {value = 0 : index} : () -> index
  %113 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%112, %98, %113) ({
  ^bb0(%arg0: index):
    "memref.store"(%111, %110, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %114 = "bufferization.to_tensor"(%110) : (memref<?xindex>) -> tensor<?xindex>
  %115 = "memref.alloc"(%30) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %116 = "arith.constant"() {value = 0 : index} : () -> index
  %117 = "arith.constant"() {value = 0 : index} : () -> index
  %118 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%117, %30, %118) ({
  ^bb0(%arg0: index):
    "memref.store"(%116, %115, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %119 = "bufferization.to_tensor"(%115) : (memref<?xindex>) -> tensor<?xindex>
  %120 = "memref.alloc"(%98) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %121 = "arith.constant"() {value = 0 : index} : () -> index
  %122 = "arith.constant"() {value = 0 : index} : () -> index
  %123 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%122, %98, %123) ({
  ^bb0(%arg0: index):
    "memref.store"(%121, %120, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %124 = "bufferization.to_tensor"(%120) : (memref<?xindex>) -> tensor<?xindex>
  %125 = "memref.alloc"(%18) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %126 = "arith.constant"() {value = 0 : index} : () -> index
  %127 = "arith.constant"() {value = 0 : index} : () -> index
  %128 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%127, %18, %128) ({
  ^bb0(%arg0: index):
    "memref.store"(%126, %125, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %129 = "bufferization.to_tensor"(%125) : (memref<?xindex>) -> tensor<?xindex>
  %130 = "memref.alloc"(%98) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %131 = "arith.constant"() {value = 0 : index} : () -> index
  %132 = "arith.constant"() {value = 0 : index} : () -> index
  %133 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%132, %98, %133) ({
  ^bb0(%arg0: index):
    "memref.store"(%131, %130, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %134 = "bufferization.to_tensor"(%130) : (memref<?xindex>) -> tensor<?xindex>
  %135 = "memref.alloc"(%22) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xindex>
  %136 = "arith.constant"() {value = 0 : index} : () -> index
  %137 = "arith.constant"() {value = 0 : index} : () -> index
  %138 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%137, %22, %138) ({
  ^bb0(%arg0: index):
    "memref.store"(%136, %135, %arg0) {nontemporal = false} : (index, memref<?xindex>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %139 = "bufferization.to_tensor"(%135) : (memref<?xindex>) -> tensor<?xindex>
  %140 = "memref.alloc"(%32) {operand_segment_sizes = array<i32: 1, 0>} : (index) -> memref<?xf64>
  %141 = "arith.constant"() {value = 0.000000e+00 : f64} : () -> f64
  %142 = "arith.constant"() {value = 0 : index} : () -> index
  %143 = "arith.constant"() {value = 1 : index} : () -> index
  "scf.for"(%142, %32, %143) ({
  ^bb0(%arg0: index):
    "memref.store"(%141, %140, %arg0) {nontemporal = false} : (f64, memref<?xf64>, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  %144 = "bufferization.to_tensor"(%140) : (memref<?xf64>) -> tensor<?xf64>
  %145 = "ta.sptensor_construct"(%104, %109, %114, %119, %124, %129, %134, %139, %144, %99, %26, %98, %30, %98, %18, %98, %22, %32, %36, %34) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>
  %146 = "arith.constant"() {value = 1 : i32} : () -> i32
  %147 = "arith.constant"() {value = 10 : i32} : () -> i32
  %148 = "memref.cast"(%41) : (memref<?xindex>) -> memref<*xindex>
  %149 = "memref.cast"(%46) : (memref<?xindex>) -> memref<*xindex>
  %150 = "memref.cast"(%51) : (memref<?xindex>) -> memref<*xindex>
  %151 = "memref.cast"(%56) : (memref<?xindex>) -> memref<*xindex>
  %152 = "memref.cast"(%61) : (memref<?xindex>) -> memref<*xindex>
  %153 = "memref.cast"(%66) : (memref<?xindex>) -> memref<*xindex>
  %154 = "memref.cast"(%71) : (memref<?xindex>) -> memref<*xf64>
  %155 = "memref.cast"(%100) : (memref<?xindex>) -> memref<*xindex>
  %156 = "memref.cast"(%105) : (memref<?xindex>) -> memref<*xindex>
  %157 = "memref.cast"(%110) : (memref<?xindex>) -> memref<*xindex>
  %158 = "memref.cast"(%115) : (memref<?xindex>) -> memref<*xindex>
  %159 = "memref.cast"(%120) : (memref<?xindex>) -> memref<*xindex>
  %160 = "memref.cast"(%125) : (memref<?xindex>) -> memref<*xindex>
  %161 = "memref.cast"(%130) : (memref<?xindex>) -> memref<*xf64>
  %162 = "memref.cast"(%6) : (memref<13xindex>) -> memref<*xindex>
  %163 = "arith.constant"() {value = -1 : index} : () -> i32
  %164 = "arith.constant"() {value = 0 : i32} : () -> i32
  %165 = "arith.constant"() {value = 1 : i32} : () -> i32
  %166 = "arith.constant"() {value = 2 : i32} : () -> i32
  %167 = "arith.constant"() {value = 3 : i32} : () -> i32
  %168 = "arith.constant"() {value = -1 : index} : () -> i32
  %169 = "arith.constant"() {value = 0 : i32} : () -> i32
  %170 = "arith.constant"() {value = 1 : i32} : () -> i32
  %171 = "arith.constant"() {value = 2 : i32} : () -> i32
  %172 = "arith.constant"() {value = 3 : i32} : () -> i32
  "func.call"(%146, %147, %166, %167, %167, %148, %149, %150, %151, %152, %153, %154, %171, %172, %172, %155, %156, %157, %158, %159, %160, %161, %162) {callee = @transpose_3D_f64} : (i32, i32, i32, i32, i32, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32, i32, i32, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, memref<*xindex>) -> ()
  "ta.print"(%145) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
  "func.return"() : () -> ()
}) {function_type = () -> (), sym_name = "main"} : () -> ()

