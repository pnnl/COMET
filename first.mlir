module {
  func.func @main() {
    %alloc = memref.alloc() : memref<19xindex>
    %cast = memref.cast %alloc : memref<19xindex> to memref<*xindex>
    %c-1 = arith.constant -1 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    call @read_input_sizes_3D_f64(%c0_i32, %c1, %c-1, %c1, %c-1, %c1, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, index, index, memref<*xindex>, i32) -> ()
    %c0 = arith.constant 0 : index
    %0 = memref.load %alloc[%c0] : memref<19xindex>
    %1 = memref.load %alloc[%c1] : memref<19xindex>
    %c2 = arith.constant 2 : index
    %2 = memref.load %alloc[%c2] : memref<19xindex>
    %c3 = arith.constant 3 : index
    %3 = memref.load %alloc[%c3] : memref<19xindex>
    %c4 = arith.constant 4 : index
    %4 = memref.load %alloc[%c4] : memref<19xindex>
    %c5 = arith.constant 5 : index
    %5 = memref.load %alloc[%c5] : memref<19xindex>
    %c6 = arith.constant 6 : index
    %6 = memref.load %alloc[%c6] : memref<19xindex>
    %c7 = arith.constant 7 : index
    %7 = memref.load %alloc[%c7] : memref<19xindex>
    %c8 = arith.constant 8 : index
    %8 = memref.load %alloc[%c8] : memref<19xindex>
    %c9 = arith.constant 9 : index
    %9 = memref.load %alloc[%c9] : memref<19xindex>
    %c10 = arith.constant 10 : index
    %10 = memref.load %alloc[%c10] : memref<19xindex>
    %c11 = arith.constant 11 : index
    %11 = memref.load %alloc[%c11] : memref<19xindex>
    %c12 = arith.constant 12 : index
    %12 = memref.load %alloc[%c12] : memref<19xindex>
    %c13 = arith.constant 13 : index
    %13 = memref.load %alloc[%c13] : memref<19xindex>
    %c14 = arith.constant 14 : index
    %14 = memref.load %alloc[%c14] : memref<19xindex>
    %c15 = arith.constant 15 : index
    %15 = memref.load %alloc[%c15] : memref<19xindex>
    %c16 = arith.constant 16 : index
    %16 = memref.load %alloc[%c16] : memref<19xindex>
    %c17 = arith.constant 17 : index
    %17 = memref.load %alloc[%c17] : memref<19xindex>
    %c18 = arith.constant 18 : index
    %18 = memref.load %alloc[%c18] : memref<19xindex>
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
    %alloc_8 = memref.alloc(%4) : memref<?xindex>
    scf.for %arg0 = %c0 to %4 step %c1 {
      memref.store %c0, %alloc_8[%arg0] : memref<?xindex>
    }
    %cast_9 = memref.cast %alloc_8 : memref<?xindex> to memref<*xindex>
    %alloc_10 = memref.alloc(%5) : memref<?xindex>
    scf.for %arg0 = %c0 to %5 step %c1 {
      memref.store %c0, %alloc_10[%arg0] : memref<?xindex>
    }
    %cast_11 = memref.cast %alloc_10 : memref<?xindex> to memref<*xindex>
    %alloc_12 = memref.alloc(%6) : memref<?xindex>
    scf.for %arg0 = %c0 to %6 step %c1 {
      memref.store %c0, %alloc_12[%arg0] : memref<?xindex>
    }
    %cast_13 = memref.cast %alloc_12 : memref<?xindex> to memref<*xindex>
    %alloc_14 = memref.alloc(%7) : memref<?xindex>
    scf.for %arg0 = %c0 to %7 step %c1 {
      memref.store %c0, %alloc_14[%arg0] : memref<?xindex>
    }
    %cast_15 = memref.cast %alloc_14 : memref<?xindex> to memref<*xindex>
    %alloc_16 = memref.alloc(%8) : memref<?xindex>
    scf.for %arg0 = %c0 to %8 step %c1 {
      memref.store %c0, %alloc_16[%arg0] : memref<?xindex>
    }
    %cast_17 = memref.cast %alloc_16 : memref<?xindex> to memref<*xindex>
    %alloc_18 = memref.alloc(%9) : memref<?xindex>
    scf.for %arg0 = %c0 to %9 step %c1 {
      memref.store %c0, %alloc_18[%arg0] : memref<?xindex>
    }
    %cast_19 = memref.cast %alloc_18 : memref<?xindex> to memref<*xindex>
    %alloc_20 = memref.alloc(%10) : memref<?xindex>
    scf.for %arg0 = %c0 to %10 step %c1 {
      memref.store %c0, %alloc_20[%arg0] : memref<?xindex>
    }
    %cast_21 = memref.cast %alloc_20 : memref<?xindex> to memref<*xindex>
    %alloc_22 = memref.alloc(%11) : memref<?xindex>
    scf.for %arg0 = %c0 to %11 step %c1 {
      memref.store %c0, %alloc_22[%arg0] : memref<?xindex>
    }
    %cast_23 = memref.cast %alloc_22 : memref<?xindex> to memref<*xindex>
    %alloc_24 = memref.alloc(%12) : memref<?xf64>
    %cst = arith.constant 0.000000e+00 : f64
    scf.for %arg0 = %c0 to %12 step %c1 {
      memref.store %cst, %alloc_24[%arg0] : memref<?xf64>
    }
    %cast_25 = memref.cast %alloc_24 : memref<?xf64> to memref<*xf64>
    call @read_input_3D_f64(%c0_i32, %c1, %c-1, %c1, %c-1, %c1, %c-1, %cast_1, %cast_3, %cast_5, %cast_7, %cast_9, %cast_11, %cast_13, %cast_15, %cast_17, %cast_19, %cast_21, %cast_23, %cast_25, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %19 = bufferization.to_tensor %alloc_0 : memref<?xindex>
    %20 = bufferization.to_tensor %alloc_2 : memref<?xindex>
    %21 = bufferization.to_tensor %alloc_4 : memref<?xindex>
    %22 = bufferization.to_tensor %alloc_6 : memref<?xindex>
    %23 = bufferization.to_tensor %alloc_8 : memref<?xindex>
    %24 = bufferization.to_tensor %alloc_10 : memref<?xindex>
    %25 = bufferization.to_tensor %alloc_12 : memref<?xindex>
    %26 = bufferization.to_tensor %alloc_14 : memref<?xindex>
    %27 = bufferization.to_tensor %alloc_16 : memref<?xindex>
    %28 = bufferization.to_tensor %alloc_18 : memref<?xindex>
    %29 = bufferization.to_tensor %alloc_20 : memref<?xindex>
    %30 = bufferization.to_tensor %alloc_22 : memref<?xindex>
    %31 = bufferization.to_tensor %alloc_24 : memref<?xf64>
    %32 = arith.addi %0, %c1 : index
    %alloc_26 = memref.alloc(%c2) : memref<?xindex>
    scf.for %arg0 = %c0 to %c2 step %c1 {
      memref.store %c0, %alloc_26[%arg0] : memref<?xindex>
    }
    %33 = bufferization.to_tensor %alloc_26 : memref<?xindex>
    %alloc_27 = memref.alloc(%0) : memref<?xindex>
    scf.for %arg0 = %c0 to %0 step %c1 {
      memref.store %c0, %alloc_27[%arg0] : memref<?xindex>
    }
    %34 = bufferization.to_tensor %alloc_27 : memref<?xindex>
    %alloc_28 = memref.alloc(%32) : memref<?xindex>
    scf.for %arg0 = %c0 to %32 step %c1 {
      memref.store %c0, %alloc_28[%arg0] : memref<?xindex>
    }
    %35 = bufferization.to_tensor %alloc_28 : memref<?xindex>
    %alloc_29 = memref.alloc(%0) : memref<?xindex>
    scf.for %arg0 = %c0 to %0 step %c1 {
      memref.store %c0, %alloc_29[%arg0] : memref<?xindex>
    }
    %36 = bufferization.to_tensor %alloc_29 : memref<?xindex>
    %alloc_30 = memref.alloc(%32) : memref<?xindex>
    scf.for %arg0 = %c0 to %32 step %c1 {
      memref.store %c0, %alloc_30[%arg0] : memref<?xindex>
    }
    %37 = bufferization.to_tensor %alloc_30 : memref<?xindex>
    %alloc_31 = memref.alloc(%0) : memref<?xindex>
    scf.for %arg0 = %c0 to %0 step %c1 {
      memref.store %c0, %alloc_31[%arg0] : memref<?xindex>
    }
    %38 = bufferization.to_tensor %alloc_31 : memref<?xindex>
    %alloc_32 = memref.alloc(%32) : memref<?xindex>
    scf.for %arg0 = %c0 to %32 step %c1 {
      memref.store %c0, %alloc_32[%arg0] : memref<?xindex>
    }
    %39 = bufferization.to_tensor %alloc_32 : memref<?xindex>
    %alloc_33 = memref.alloc(%0) : memref<?xindex>
    scf.for %arg0 = %c0 to %0 step %c1 {
      memref.store %c0, %alloc_33[%arg0] : memref<?xindex>
    }
    %40 = bufferization.to_tensor %alloc_33 : memref<?xindex>
    %alloc_34 = memref.alloc(%32) : memref<?xindex>
    scf.for %arg0 = %c0 to %32 step %c1 {
      memref.store %c0, %alloc_34[%arg0] : memref<?xindex>
    }
    %41 = bufferization.to_tensor %alloc_34 : memref<?xindex>
    %alloc_35 = memref.alloc(%0) : memref<?xindex>
    scf.for %arg0 = %c0 to %0 step %c1 {
      memref.store %c0, %alloc_35[%arg0] : memref<?xindex>
    }
    %42 = bufferization.to_tensor %alloc_35 : memref<?xindex>
    %alloc_36 = memref.alloc(%32) : memref<?xindex>
    scf.for %arg0 = %c0 to %32 step %c1 {
      memref.store %c0, %alloc_36[%arg0] : memref<?xindex>
    }
    %43 = bufferization.to_tensor %alloc_36 : memref<?xindex>
    %alloc_37 = memref.alloc(%0) : memref<?xindex>
    scf.for %arg0 = %c0 to %0 step %c1 {
      memref.store %c0, %alloc_37[%arg0] : memref<?xindex>
    }
    %44 = bufferization.to_tensor %alloc_37 : memref<?xindex>
    %alloc_38 = memref.alloc(%32) : memref<?xf64>
    scf.for %arg0 = %c0 to %32 step %c1 {
      memref.store %cst, %alloc_38[%arg0] : memref<?xf64>
    }
    %45 = bufferization.to_tensor %alloc_38 : memref<?xf64>
    %c12_i32 = arith.constant 12 : i32
    %c201_i32 = arith.constant 201 : i32
    %cast_39 = memref.cast %alloc_26 : memref<?xindex> to memref<*xindex>
    %cast_40 = memref.cast %alloc_27 : memref<?xindex> to memref<*xindex>
    %cast_41 = memref.cast %alloc_28 : memref<?xindex> to memref<*xindex>
    %cast_42 = memref.cast %alloc_29 : memref<?xindex> to memref<*xindex>
    %cast_43 = memref.cast %alloc_30 : memref<?xindex> to memref<*xindex>
    %cast_44 = memref.cast %alloc_31 : memref<?xindex> to memref<*xindex>
    %cast_45 = memref.cast %alloc_32 : memref<?xindex> to memref<*xindex>
    %cast_46 = memref.cast %alloc_33 : memref<?xindex> to memref<*xindex>
    %cast_47 = memref.cast %alloc_34 : memref<?xindex> to memref<*xindex>
    %cast_48 = memref.cast %alloc_35 : memref<?xindex> to memref<*xindex>
    %cast_49 = memref.cast %alloc_36 : memref<?xindex> to memref<*xindex>
    %cast_50 = memref.cast %alloc_37 : memref<?xindex> to memref<*xindex>
    %cast_51 = memref.cast %alloc_38 : memref<?xf64> to memref<*xf64>
    %c-1_i32 = arith.constant -1 : i32
    call @transpose_3D_f64(%c12_i32, %c201_i32, %c1_i32, %c-1_i32, %c1_i32, %c-1_i32, %c1_i32, %c-1_i32, %cast_1, %cast_3, %cast_5, %cast_7, %cast_9, %cast_11, %cast_13, %cast_15, %cast_17, %cast_19, %cast_21, %cast_23, %cast_25, %c1_i32, %c-1_i32, %c1_i32, %c-1_i32, %c1_i32, %c-1_i32, %cast_39, %cast_40, %cast_41, %cast_42, %cast_43, %cast_44, %cast_45, %cast_46, %cast_47, %cast_48, %cast_49, %cast_50, %cast_51, %cast) : (i32, i32, i32, i32, i32, i32, i32, i32, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32, i32, i32, i32, i32, i32, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_39) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_40) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_41) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_42) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_43) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_44) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_45) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_46) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_47) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_48) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_49) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_50) : (memref<*xindex>) -> ()
    call @comet_print_memref_f64(%cast_51) : (memref<*xf64>) -> ()
    return
  }
  func.func private @read_input_3D_f64(i32, index, index, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_3D_f64(i32, index, index, index, index, index, index, memref<*xindex>, i32)
  func.func private @transpose_3D_f64(i32, i32, i32, i32, i32, i32, i32, i32, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32, i32, i32, i32, i32, i32, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, memref<*xindex>)
  func.func private @quick_sort(memref<*xindex>, index)
  func.func private @comet_print_memref_f64(memref<*xf64>)
  func.func private @comet_print_memref_i64(memref<*xindex>)
}
