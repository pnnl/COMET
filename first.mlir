module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = "ta.dynamic_index_label"(%c0, %c1) : (index, index) -> !ta.range
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
    %alloc = memref.alloc() : memref<7xindex>
    %cast = memref.cast %alloc : memref<7xindex> to memref<*xindex>
    %c-1 = arith.constant -1 : index
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_4 = arith.constant 1 : i32
    call @read_input_sizes_2D_f64(%c1_i32, %c0_2, %c3, %c0_2, %c-1, %cast, %c1_i32_4) {filename = "SPARSE_FILE_NAME1"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
    %c0_5 = arith.constant 0 : index
    %2 = memref.load %alloc[%c0_5] : memref<7xindex>
    %c1_6 = arith.constant 1 : index
    %3 = memref.load %alloc[%c1_6] : memref<7xindex>
    %c2_7 = arith.constant 2 : index
    %4 = memref.load %alloc[%c2_7] : memref<7xindex>
    %c3_8 = arith.constant 3 : index
    %5 = memref.load %alloc[%c3_8] : memref<7xindex>
    %c4 = arith.constant 4 : index
    %6 = memref.load %alloc[%c4] : memref<7xindex>
    %c5 = arith.constant 5 : index
    %7 = memref.load %alloc[%c5] : memref<7xindex>
    %c6 = arith.constant 6 : index
    %8 = memref.load %alloc[%c6] : memref<7xindex>
    %c7 = arith.constant 7 : index
    %9 = memref.load %alloc[%c7] : memref<7xindex>
    %c8 = arith.constant 8 : index
    %10 = memref.load %alloc[%c8] : memref<7xindex>
    %c9 = arith.constant 9 : index
    %11 = memref.load %alloc[%c9] : memref<7xindex>
    %c10 = arith.constant 10 : index
    %12 = memref.load %alloc[%c10] : memref<7xindex>
    %alloc_9 = memref.alloc(%2) : memref<?xindex>
    %c0_10 = arith.constant 0 : index
    %c0_11 = arith.constant 0 : index
    %c1_12 = arith.constant 1 : index
    scf.for %arg0 = %c0_11 to %2 step %c1_12 {
      memref.store %c0_10, %alloc_9[%arg0] : memref<?xindex>
    }
    %cast_13 = memref.cast %alloc_9 : memref<?xindex> to memref<*xindex>
    %alloc_14 = memref.alloc(%3) : memref<?xindex>
    %c0_15 = arith.constant 0 : index
    %c0_16 = arith.constant 0 : index
    %c1_17 = arith.constant 1 : index
    scf.for %arg0 = %c0_16 to %3 step %c1_17 {
      memref.store %c0_15, %alloc_14[%arg0] : memref<?xindex>
    }
    %cast_18 = memref.cast %alloc_14 : memref<?xindex> to memref<*xindex>
    %alloc_19 = memref.alloc(%4) : memref<?xindex>
    %c0_20 = arith.constant 0 : index
    %c0_21 = arith.constant 0 : index
    %c1_22 = arith.constant 1 : index
    scf.for %arg0 = %c0_21 to %4 step %c1_22 {
      memref.store %c0_20, %alloc_19[%arg0] : memref<?xindex>
    }
    %cast_23 = memref.cast %alloc_19 : memref<?xindex> to memref<*xindex>
    %alloc_24 = memref.alloc(%5) : memref<?xindex>
    %c0_25 = arith.constant 0 : index
    %c0_26 = arith.constant 0 : index
    %c1_27 = arith.constant 1 : index
    scf.for %arg0 = %c0_26 to %5 step %c1_27 {
      memref.store %c0_25, %alloc_24[%arg0] : memref<?xindex>
    }
    %cast_28 = memref.cast %alloc_24 : memref<?xindex> to memref<*xindex>
    %alloc_29 = memref.alloc(%6) : memref<?xindex>
    %c0_30 = arith.constant 0 : index
    %c0_31 = arith.constant 0 : index
    %c1_32 = arith.constant 1 : index
    scf.for %arg0 = %c0_31 to %6 step %c1_32 {
      memref.store %c0_30, %alloc_29[%arg0] : memref<?xindex>
    }
    %cast_33 = memref.cast %alloc_29 : memref<?xindex> to memref<*xindex>
    %alloc_34 = memref.alloc(%7) : memref<?xindex>
    %c0_35 = arith.constant 0 : index
    %c0_36 = arith.constant 0 : index
    %c1_37 = arith.constant 1 : index
    scf.for %arg0 = %c0_36 to %7 step %c1_37 {
      memref.store %c0_35, %alloc_34[%arg0] : memref<?xindex>
    }
    %cast_38 = memref.cast %alloc_34 : memref<?xindex> to memref<*xindex>
    %alloc_39 = memref.alloc(%8) : memref<?xindex>
    %c0_40 = arith.constant 0 : index
    %c0_41 = arith.constant 0 : index
    %c1_42 = arith.constant 1 : index
    scf.for %arg0 = %c0_41 to %8 step %c1_42 {
      memref.store %c0_40, %alloc_39[%arg0] : memref<?xindex>
    }
    %cast_43 = memref.cast %alloc_39 : memref<?xindex> to memref<*xindex>
    %alloc_44 = memref.alloc(%9) : memref<?xindex>
    %c0_45 = arith.constant 0 : index
    %c0_46 = arith.constant 0 : index
    %c1_47 = arith.constant 1 : index
    scf.for %arg0 = %c0_46 to %9 step %c1_47 {
      memref.store %c0_45, %alloc_44[%arg0] : memref<?xindex>
    }
    %cast_48 = memref.cast %alloc_44 : memref<?xindex> to memref<*xindex>
    %alloc_49 = memref.alloc(%10) : memref<?xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %c0_50 = arith.constant 0 : index
    %c1_51 = arith.constant 1 : index
    scf.for %arg0 = %c0_50 to %10 step %c1_51 {
      memref.store %cst, %alloc_49[%arg0] : memref<?xf64>
    }
    %cast_52 = memref.cast %alloc_49 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c1_i32, %c0_2, %c3, %c0_2, %c-1, %cast_13, %cast_18, %cast_23, %cast_28, %cast_33, %cast_38, %cast_43, %cast_48, %cast_52, %c1_i32_4) {filename = "SPARSE_FILE_NAME1"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %13 = bufferization.to_tensor %alloc_9 : memref<?xindex>
    %14 = bufferization.to_tensor %alloc_14 : memref<?xindex>
    %15 = bufferization.to_tensor %alloc_19 : memref<?xindex>
    %16 = bufferization.to_tensor %alloc_24 : memref<?xindex>
    %17 = bufferization.to_tensor %alloc_29 : memref<?xindex>
    %18 = bufferization.to_tensor %alloc_34 : memref<?xindex>
    %19 = bufferization.to_tensor %alloc_39 : memref<?xindex>
    %20 = bufferization.to_tensor %alloc_44 : memref<?xindex>
    %21 = bufferization.to_tensor %alloc_49 : memref<?xf64>
    %22 = ta.sptensor_construct(%13, %14, %15, %16, %17, %18, %19, %20, %21, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
    %23 = "ta.static_index_label"(%c0, %11, %c1) : (index, index, index) -> !ta.range
    %24 = "ta.static_index_label"(%c0_0, %12, %c1_1) : (index, index, index) -> !ta.range
    "ta.print"(%22) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
}
