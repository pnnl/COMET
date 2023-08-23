module  {
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %c5 = arith.constant 5 : index
    %c1_i32 = arith.constant 1 : i32
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %cst = arith.constant 0.000000e+00 : f64
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<7xindex>
    %1 = memref.cast %0 : memref<7xindex> to memref<*xindex>
    call @read_input_sizes_2D_f64(%c0_i32, %c0, %c1, %1) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, memref<*xindex>) -> ()
    %2 = memref.load %0[%c0] : memref<7xindex>
    %3 = memref.load %0[%c1] : memref<7xindex>
    %4 = memref.load %0[%c2] : memref<7xindex>
    %5 = memref.load %0[%c3] : memref<7xindex>
    %6 = memref.load %0[%c4] : memref<7xindex>
    %7 = memref.load %0[%c5] : memref<7xindex>  // num_rows
    %8 = memref.alloc(%2) : memref<?xindex>
    scf.for %arg0 = %c0 to %2 step %c1 {
      memref.store %c0, %8[%arg0] : memref<?xindex>
    }
    %9 = memref.cast %8 : memref<?xindex> to memref<*xindex>
    %10 = memref.alloc(%3) : memref<?xindex>
    scf.for %arg0 = %c0 to %3 step %c1 {
      memref.store %c0, %10[%arg0] : memref<?xindex>
    }
    %11 = memref.cast %10 : memref<?xindex> to memref<*xindex>
    %12 = memref.alloc(%4) : memref<?xindex>
    scf.for %arg0 = %c0 to %4 step %c1 {
      memref.store %c0, %12[%arg0] : memref<?xindex>
    }
    %13 = memref.cast %12 : memref<?xindex> to memref<*xindex>
    %14 = memref.alloc(%5) : memref<?xindex>
    scf.for %arg0 = %c0 to %5 step %c1 {
      memref.store %c0, %14[%arg0] : memref<?xindex>
    }
    %15 = memref.cast %14 : memref<?xindex> to memref<*xindex>
    %16 = memref.alloc(%6) : memref<?xf64>
    scf.for %arg0 = %c0 to %6 step %c1 {
      memref.store %cst, %16[%arg0] : memref<?xf64>
    }
    %17 = memref.cast %16 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c0_i32, %c0, %c1, %9, %11, %13, %15, %17) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>) -> ()
    %18 = memref.alloc() : memref<7xindex>
    %19 = memref.cast %18 : memref<7xindex> to memref<*xindex>
    call @read_input_sizes_2D_f64(%c1_i32, %c0, %c1, %19) {filename = "SPARSE_FILE_NAME1"} : (i32, index, index, memref<*xindex>) -> ()
    %20 = memref.load %18[%c0] : memref<7xindex>  // 1
    %21 = memref.load %18[%c1] : memref<7xindex>  // 1
    %22 = memref.load %18[%c2] : memref<7xindex>  // num_rows + 1
    %23 = memref.load %18[%c3] : memref<7xindex>  // num_non_zeros
    %24 = memref.load %18[%c4] : memref<7xindex>  // num_non_zeros
    %25 = memref.load %18[%c6] : memref<7xindex>  // num_cols
    %26 = memref.alloc(%20) : memref<?xindex>
    scf.for %arg0 = %c0 to %20 step %c1 {
      memref.store %c0, %26[%arg0] : memref<?xindex>
    }
    %27 = memref.cast %26 : memref<?xindex> to memref<*xindex>
    %28 = memref.alloc(%21) : memref<?xindex>
    scf.for %arg0 = %c0 to %21 step %c1 {
      memref.store %c0, %28[%arg0] : memref<?xindex>
    }
    %29 = memref.cast %28 : memref<?xindex> to memref<*xindex>
    %30 = memref.alloc(%22) : memref<?xindex>
    scf.for %arg0 = %c0 to %22 step %c1 {
      memref.store %c0, %30[%arg0] : memref<?xindex>
    }
    %31 = memref.cast %30 : memref<?xindex> to memref<*xindex>
    %32 = memref.alloc(%23) : memref<?xindex>
    scf.for %arg0 = %c0 to %23 step %c1 {
      memref.store %c0, %32[%arg0] : memref<?xindex>
    }
    %33 = memref.cast %32 : memref<?xindex> to memref<*xindex>
    %34 = memref.alloc(%24) : memref<?xf64>
    scf.for %arg0 = %c0 to %24 step %c1 {
      memref.store %cst, %34[%arg0] : memref<?xf64>
    }
    %35 = memref.cast %34 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c1_i32, %c0, %c1, %27, %29, %31, %33, %35) {filename = "SPARSE_FILE_NAME1"} : (i32, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>) -> ()
    %36 = memref.alloc(%25) {alignment = 32 : i64} : memref<?xf64>
    scf.for %arg0 = %c0 to %25 step %c1 {
      memref.store %cst, %36[%arg0] : memref<?xf64>
    }
    %37 = memref.alloc(%25) {alignment = 32 : i64} : memref<?xi1>
    scf.for %arg0 = %c0 to %25 step %c1 {
      memref.store %false, %37[%arg0] : memref<?xi1>
    }
    %38 = memref.alloc(%25) {alignment = 32 : i64} : memref<?xindex>
    scf.for %arg0 = %c0 to %25 step %c1 {
      memref.store %c0, %38[%arg0] : memref<?xindex>
    }
    %39 = memref.alloc() : memref<1xindex>
    %40 = arith.addi %7, %c1 : index
    // %41 = arith.muli %7, %25 : index  // %41 = num_rows * num_cols

    // %41 = arith.constant 40000000 : index
    %41 = arith.constant 2000000000 : index

    %42 = memref.alloc(%c1) : memref<?xindex>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      memref.store %c0, %42[%arg0] : memref<?xindex>
    }
    %43 = memref.alloc(%c1) : memref<?xindex>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      memref.store %c0, %43[%arg0] : memref<?xindex>
    }
    %44 = memref.alloc(%40) : memref<?xindex>
    scf.for %arg0 = %c0 to %40 step %c1 {
      memref.store %c0, %44[%arg0] : memref<?xindex>
    }

    %45 = memref.alloc(%41) : memref<?xindex>
    scf.for %arg0 = %c0 to %41 step %c1 {
      memref.store %c0, %45[%arg0] : memref<?xindex>
    }

    %46 = memref.alloc(%41) : memref<?xf64>
    scf.for %arg0 = %c0 to %41 step %c1 {
      memref.store %cst, %46[%arg0] : memref<?xf64>
    }

    %47 = memref.alloc() : memref<1xindex>
    memref.store %c1, %47[%c0] : memref<1xindex>
    %48 = memref.alloc() : memref<1xindex>
    memref.store %c1, %48[%c0] : memref<1xindex>
    %49 = memref.alloc() : memref<1xindex>
    memref.store %c1, %49[%c0] : memref<1xindex>
    %50 = memref.alloc() : memref<1xindex>
    memref.store %c0, %50[%c0] : memref<1xindex>
    %51 = memref.alloc() : memref<1xindex>
    memref.store %c0, %51[%c0] : memref<1xindex>
    memref.store %7, %42[%c0] : memref<?xindex>
    %52 = memref.load %8[%c0] : memref<?xindex>
    %53 = memref.alloc(%25) : memref<?xi1>
    scf.for %arg0 = %c0 to %25 step %c1 {
      memref.store %false, %53[%arg0] : memref<?xi1>
    }

    %time_start = call @getTime() : () -> f64

    scf.for %arg0 = %c0 to %52 step %c1 {
      memref.store %c0, %39[%c0] : memref<1xindex>
      %59 = arith.addi %arg0, %c1 : index
      %60 = memref.load %12[%arg0] : memref<?xindex>
      %61 = memref.load %12[%59] : memref<?xindex>
      %62 = memref.load %30[%arg0] : memref<?xindex>
      %63 = memref.load %30[%59] : memref<?xindex>
      scf.for %arg1 = %62 to %63 step %c1 {
        %69 = memref.load %34[%arg1] : memref<?xf64>
        %70 = arith.cmpf une, %69, %cst : f64
        scf.if %70 {
          %71 = memref.load %32[%arg1] : memref<?xindex>
          memref.store %true, %53[%71] : memref<?xi1>
        }
      }
      scf.for %arg1 = %60 to %61 step %c1 {
        %69 = memref.load %14[%arg1] : memref<?xindex>
        %70 = arith.addi %69, %c1 : index
        %71 = memref.load %30[%69] : memref<?xindex>
        %72 = memref.load %30[%70] : memref<?xindex>
        scf.for %arg2 = %71 to %72 step %c1 {
          %73 = memref.load %32[%arg2] : memref<?xindex>  /// B.col = %32
          %74 = memref.load %53[%73] : memref<?xi1>
          %75 = arith.cmpi eq, %74, %true : i1
          scf.if %75 {
            %76 = memref.load %37[%73] : memref<?xi1>
            %77 = arith.cmpi eq, %76, %false : i1
            scf.if %77 {
              %78 = memref.load %16[%arg1] : memref<?xf64>
              %79 = memref.load %34[%arg2] : memref<?xf64>
              %80 = arith.mulf %78, %79 : f64
              memref.store %80, %36[%73] : memref<?xf64>
              memref.store %true, %37[%73] : memref<?xi1>
              %81 = memref.load %39[%c0] : memref<1xindex>
              memref.store %73, %38[%81] : memref<?xindex>
              %82 = arith.addi %81, %c1 : index
              memref.store %82, %39[%c0] : memref<1xindex>
            } else {
              %78 = memref.load %16[%arg1] : memref<?xf64>
              %79 = memref.load %34[%arg2] : memref<?xf64>
              %80 = memref.load %36[%73] : memref<?xf64>
              %81 = arith.mulf %78, %79 : f64
              %82 = arith.addf %80, %81 : f64
              memref.store %82, %36[%73] : memref<?xf64>
            }
          }
        }
      }
      scf.for %arg1 = %62 to %63 step %c1 {
        %69 = memref.load %32[%arg1] : memref<?xindex>
        memref.store %false, %53[%69] : memref<?xi1>
      }
      %64 = memref.load %39[%c0] : memref<1xindex>
      %65 = memref.cast %38 : memref<?xindex> to memref<*xindex>
      func.call @quick_sort(%65, %64) : (memref<*xindex>, index) -> ()
      scf.for %arg1 = %c0 to %64 step %c1 {
        %69 = memref.load %51[%c0] : memref<1xindex>
        %70 = memref.load %38[%arg1] : memref<?xindex>
        %71 = memref.load %36[%70] : memref<?xf64>
        memref.store %71, %46[%69] : memref<?xf64>   /// C.val = %46
        memref.store %false, %37[%70] : memref<?xi1>
        memref.store %70, %45[%69] : memref<?xindex>   /// C.col = %45
        %72 = arith.addi %69, %c1 : index
        memref.store %72, %51[%c0] : memref<1xindex>
        memref.store %72, %50[%c0] : memref<1xindex>
      }
      %66 = memref.load %49[%c0] : memref<1xindex>
      %67 = memref.load %50[%c0] : memref<1xindex>
      memref.store %67, %44[%66] : memref<?xindex>
      %68 = arith.addi %66, %c1 : index
      memref.store %68, %49[%c0] : memref<1xindex>
    }
    %time_end = call @getTime() : () -> f64
    call @printElapsedTime(%time_start, %time_end) : (f64, f64) -> ()
    // %54 = memref.cast %42 : memref<?xindex> to memref<*xindex>
    // call @comet_print_memref_i64(%54) : (memref<*xindex>) -> ()
    // %55 = memref.cast %43 : memref<?xindex> to memref<*xindex>
    // call @comet_print_memref_i64(%55) : (memref<*xindex>) -> ()
    // %56 = memref.cast %44 : memref<?xindex> to memref<*xindex>
    // call @comet_print_memref_i64(%56) : (memref<*xindex>) -> ()
    // %57 = memref.cast %45 : memref<?xindex> to memref<*xindex>
    // call @comet_print_memref_i64(%57) : (memref<*xindex>) -> ()
    // %58 = memref.cast %46 : memref<?xf64> to memref<*xf64>
    // call @comet_print_memref_f64(%58) : (memref<*xf64>) -> ()

    %88 = memref.alloc() : memref<1xindex>
    memref.store %c0, %88[%c0] : memref<1xindex>
    %89 = memref.load %51[%c0] : memref<1xindex>  // value size = %51
    scf.for %arg0 = %c0 to %89 step %c1 {
      %91 = memref.load %46[%arg0] : memref<?xf64>  // output.val = %46
      %191 = arith.fptoui %91 : f64 to i64
      %291 = arith.index_cast %191 : i64 to index
      %92 = memref.load %88[%c0] : memref<1xindex>
      %93 = arith.addi %291, %92 : index
      memref.store %93, %88[%c0] : memref<1xindex>
    }
    %90 = memref.cast %88 : memref<1xindex> to memref<*xindex>
    call @comet_print_memref_i64(%90) : (memref<*xindex>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>)
  func.func private @read_input_sizes_2D_f64(i32, index, index, memref<*xindex>)
  func.func private @quick_sort(memref<*xindex>, index)
  func.func private @getTime() -> f64
  func.func private @printElapsedTime(f64, f64)
  func.func private @comet_print_memref_f64(memref<*xf64>)
  func.func private @comet_print_memref_i64(memref<*xindex>)
}
