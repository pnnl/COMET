module {
  func.func @main() {
    %alloc = memref.alloc() : memref<7xindex>
    %cast = memref.cast %alloc : memref<7xindex> to memref<*xindex>
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1_i32 = arith.constant 1 : i32
    call @read_input_sizes_2D_f64(%c1_i32, %c0, %c3, %c0, %c-1, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME1"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
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
    call @read_input_2D_f64(%c1_i32, %c0, %c3, %c0, %c-1, %cast_1, %cast_3, %cast_5, %cast_7, %cast_9, %c1_i32) {filename = "SPARSE_FILE_NAME1"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
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
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @comet_print_memref_f64(memref<*xf64>)
  func.func private @comet_print_memref_i64(memref<*xindex>)
}
