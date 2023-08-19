module {
  func.func @main() {
    %alloc = memref.alloc() : memref<13xindex>
    %cast = memref.cast %alloc : memref<13xindex> to memref<*xindex>
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    call @read_input_sizes_2D_f64(%c0_i32, %c0, %c0, %c3, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
    %0 = memref.load %alloc[%c0] : memref<13xindex>
    %c1 = arith.constant 1 : index
    %1 = memref.load %alloc[%c1] : memref<13xindex>
    %c2 = arith.constant 2 : index
    %2 = memref.load %alloc[%c2] : memref<13xindex>
    %3 = memref.load %alloc[%c3] : memref<13xindex>
    %c4 = arith.constant 4 : index
    %4 = memref.load %alloc[%c4] : memref<13xindex>
    %c5 = arith.constant 5 : index
    %5 = memref.load %alloc[%c5] : memref<13xindex>
    %c6 = arith.constant 6 : index
    %6 = memref.load %alloc[%c6] : memref<13xindex>
    %c7 = arith.constant 7 : index
    %7 = memref.load %alloc[%c7] : memref<13xindex>
    %c8 = arith.constant 8 : index
    %8 = memref.load %alloc[%c8] : memref<13xindex>
    %c9 = arith.constant 9 : index
    %9 = memref.load %alloc[%c9] : memref<13xindex>
    %c10 = arith.constant 10 : index
    %10 = memref.load %alloc[%c10] : memref<13xindex>
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
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
    %alloc_16 = memref.alloc(%8) : memref<?xf64>
    %cst = arith.constant 0.000000e+00 : f64
    scf.for %arg0 = %c0 to %8 step %c1 {
      memref.store %cst, %alloc_16[%arg0] : memref<?xf64>
    }
    %cast_17 = memref.cast %alloc_16 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c0_i32, %c0, %c0, %c3, %c-1, %cast_1, %cast_3, %cast_5, %cast_7, %cast_9, %cast_11, %cast_13, %cast_15, %cast_17, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %11 = bufferization.to_tensor %alloc_0 : memref<?xindex>
    %12 = bufferization.to_tensor %alloc_2 : memref<?xindex>
    %13 = bufferization.to_tensor %alloc_4 : memref<?xindex>
    %14 = bufferization.to_tensor %alloc_6 : memref<?xindex>
    %15 = bufferization.to_tensor %alloc_8 : memref<?xindex>
    %16 = bufferization.to_tensor %alloc_10 : memref<?xindex>
    %17 = bufferization.to_tensor %alloc_12 : memref<?xindex>
    %18 = bufferization.to_tensor %alloc_14 : memref<?xindex>
    %19 = bufferization.to_tensor %alloc_16 : memref<?xf64>
    %alloc_18 = memref.alloc(%10) {alignment = 32 : i64} : memref<?xf64>
    %alloc_19 = memref.alloc(%9) {alignment = 32 : i64} : memref<?xf64>
    %cst_20 = arith.constant 1.700000e+00 : f64
    linalg.fill ins(%cst_20 : f64) outs(%alloc_18 : memref<?xf64>)
    linalg.fill ins(%cst : f64) outs(%alloc_19 : memref<?xf64>)
    %20 = memref.load %alloc_0[%c0] : memref<?xindex>
    scf.for %arg0 = %c0 to %20 step %c1 {
      %21 = memref.load %alloc_8[%c0] : memref<?xindex>
      scf.for %arg1 = %c0 to %21 step %c1 {
        %22 = memref.load %alloc_16[%arg1] : memref<?xf64>
        %23 = memref.load %alloc_18[%arg1] : memref<?xf64>
        %24 = memref.load %alloc_19[%arg0] : memref<?xf64>
        %25 = arith.mulf %22, %23 : f64
        %26 = arith.addf %24, %25 : f64
        memref.store %26, %alloc_19[%arg0] : memref<?xf64>
      }
    }
    %cast_21 = memref.cast %alloc_19 : memref<?xf64> to memref<*xf64>
    call @comet_print_memref_f64(%cast_21) : (memref<*xf64>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @quick_sort(memref<*xindex>, index)
  func.func private @comet_print_memref_f64(memref<*xf64>)
}
