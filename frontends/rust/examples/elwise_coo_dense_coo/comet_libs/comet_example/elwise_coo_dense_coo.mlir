module  {
  func @elwise_coo_dense_coo() {
    %c2_i32 = constant 2 : i32
    %c1_i32 = constant 1 : i32
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c4 = constant 4 : index
    %c5 = constant 5 : index
    %c6 = constant 6 : index
    %cst = constant 2.700000e+00 : f64
    %c0 = constant 0 : index
    %cst_0 = constant 0.000000e+00 : f64
    %c1 = constant 1 : index
    %0 = memref.alloc() : memref<7xindex>
    %1 = memref.cast %0 : memref<7xindex> to memref<*xindex>
    call @read_input_sizes_2D_f64(%c2_i32, %c2, %c3, %1, %c1_i32) {filename = "SPARSE_FILE_NAME2"} : (i32, index, index, memref<*xindex>, i32) -> ()
    %2 = memref.load %0[%c0] : memref<7xindex>
    %3 = memref.load %0[%c1] : memref<7xindex>
    %4 = memref.load %0[%c2] : memref<7xindex>
    %5 = memref.load %0[%c3] : memref<7xindex>
    %6 = memref.load %0[%c4] : memref<7xindex>
    %7 = memref.load %0[%c5] : memref<7xindex>
    %8 = memref.load %0[%c6] : memref<7xindex>
    %9 = memref.alloc(%2) : memref<?xindex>
    scf.for %arg0 = %c0 to %2 step %c1 {
      memref.store %c0, %9[%arg0] : memref<?xindex>
    }
    %10 = memref.cast %9 : memref<?xindex> to memref<*xindex>
    %11 = memref.alloc(%3) : memref<?xindex>
    scf.for %arg0 = %c0 to %3 step %c1 {
      memref.store %c0, %11[%arg0] : memref<?xindex>
    }
    %12 = memref.cast %11 : memref<?xindex> to memref<*xindex>
    %13 = memref.alloc(%4) : memref<?xindex>
    scf.for %arg0 = %c0 to %4 step %c1 {
      memref.store %c0, %13[%arg0] : memref<?xindex>
    }
    %14 = memref.cast %13 : memref<?xindex> to memref<*xindex>
    %15 = memref.alloc(%5) : memref<?xindex>
    scf.for %arg0 = %c0 to %5 step %c1 {
      memref.store %c0, %15[%arg0] : memref<?xindex>
    }
    %16 = memref.cast %15 : memref<?xindex> to memref<*xindex>
    %17 = memref.alloc(%6) : memref<?xf64>
    scf.for %arg0 = %c0 to %6 step %c1 {
      memref.store %cst_0, %17[%arg0] : memref<?xf64>
    }
    %18 = memref.cast %17 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c2_i32, %c2, %c3, %10, %12, %14, %16, %18, %c1_i32) {filename = "SPARSE_FILE_NAME2"} : (i32, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %19 = memref.alloc(%7, %8) {alignment = 32 : i64} : memref<?x?xf64>
    scf.for %arg0 = %c0 to %7 step %c1 {
      scf.for %arg1 = %c0 to %8 step %c1 {
        memref.store %cst, %19[%arg0, %arg1] : memref<?x?xf64>
      }
    }
    %20 = memref.alloc(%2) : memref<?xindex>
    scf.for %arg0 = %c0 to %2 step %c1 {
      memref.store %c0, %20[%arg0] : memref<?xindex>
    }
    %21 = memref.alloc(%3) : memref<?xindex>
    scf.for %arg0 = %c0 to %3 step %c1 {
      memref.store %c0, %21[%arg0] : memref<?xindex>
    }
    %22 = memref.alloc(%4) : memref<?xindex>
    scf.for %arg0 = %c0 to %4 step %c1 {
      memref.store %c0, %22[%arg0] : memref<?xindex>
    }
    %23 = memref.alloc(%5) : memref<?xindex>
    scf.for %arg0 = %c0 to %5 step %c1 {
      memref.store %c0, %23[%arg0] : memref<?xindex>
    }
    %24 = memref.alloc(%6) : memref<?xf64>
    scf.for %arg0 = %c0 to %6 step %c1 {
      memref.store %cst_0, %24[%arg0] : memref<?xf64>
    }
    %25 = memref.load %9[%c0] : memref<?xindex>
    %26 = memref.load %9[%c1] : memref<?xindex>
    %27 = memref.alloc() : memref<1xindex>
    memref.store %c0, %27[%c0] : memref<1xindex>
    scf.for %arg0 = %25 to %26 step %c1 {
      %33 = memref.load %11[%arg0] : memref<?xindex>
      %34 = memref.load %15[%arg0] : memref<?xindex>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        %36 = memref.load %17[%arg0] : memref<?xf64>
        %37 = memref.load %19[%33, %34] : memref<?x?xf64>
        %38 = cmpf one, %37, %cst_0 : f64
        scf.if %38 {
          %39 = mulf %36, %37 : f64
          %40 = memref.load %27[%c0] : memref<1xindex>
          memref.store %39, %24[%40] : memref<?xf64>
          memref.store %33, %21[%40] : memref<?xindex>
          memref.store %34, %23[%40] : memref<?xindex>
          %41 = addi %40, %c1 : index
          memref.store %41, %27[%c0] : memref<1xindex>
        }
      }
      %35 = memref.load %27[%c0] : memref<1xindex>
      memref.store %c0, %20[%c0] : memref<?xindex>
      memref.store %35, %20[%c1] : memref<?xindex>
    }
    %28 = memref.cast %20 : memref<?xindex> to memref<*xindex>
    call @comet_print_memref_i64(%28) : (memref<*xindex>) -> ()
    %29 = memref.cast %21 : memref<?xindex> to memref<*xindex>
    call @comet_print_memref_i64(%29) : (memref<*xindex>) -> ()
    %30 = memref.cast %22 : memref<?xindex> to memref<*xindex>
    call @comet_print_memref_i64(%30) : (memref<*xindex>) -> ()
    %31 = memref.cast %23 : memref<?xindex> to memref<*xindex>
    call @comet_print_memref_i64(%31) : (memref<*xindex>) -> ()
    %32 = memref.cast %24 : memref<?xf64> to memref<*xf64>
    call @comet_print_memref_f64(%32) : (memref<*xf64>) -> ()
    return
  }
  func private @read_input_2D_f64(i32, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func private @read_input_sizes_2D_f64(i32, index, index, memref<*xindex>, i32)
  func private @quick_sort(memref<*xindex>, index)
  func private @comet_print_memref_f64(memref<*xf64>)
  func private @comet_print_memref_i64(memref<*xindex>)
}
