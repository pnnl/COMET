module  {
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %c5 = arith.constant 5 : index
    %c1_i32 = arith.constant 1 : i32
    %c6 = arith.constant 6 : index
    %c2_i32 = arith.constant 2 : i32
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<7xindex>
    %1 = memref.cast %0 : memref<7xindex> to memref<*xindex>
    func.call @read_input_sizes_2D_f64(%c0_i32, %c0, %c1, %1, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, memref<*xindex>, i32) -> ()
    %2 = memref.load %0[%c0] : memref<7xindex>
    %3 = memref.load %0[%c1] : memref<7xindex>
    %4 = memref.load %0[%c2] : memref<7xindex>
    %5 = memref.load %0[%c3] : memref<7xindex>
    %6 = memref.load %0[%c4] : memref<7xindex>
    %7 = memref.load %0[%c5] : memref<7xindex>
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
    %12 = memref.alloc(%4) : memref<?xindex> // %12 = A.rowptr
    scf.for %arg0 = %c0 to %4 step %c1 {
      memref.store %c0, %12[%arg0] : memref<?xindex>
    }
    %13 = memref.cast %12 : memref<?xindex> to memref<*xindex>
    %14 = memref.alloc(%5) : memref<?xindex> // %14 = A.col
    scf.for %arg0 = %c0 to %5 step %c1 {
      memref.store %c0, %14[%arg0] : memref<?xindex>
    }
    %15 = memref.cast %14 : memref<?xindex> to memref<*xindex>
    %16 = memref.alloc(%6) : memref<?xf64> // %16 = A.val
    scf.for %arg0 = %c0 to %6 step %c1 {
      memref.store %cst, %16[%arg0] : memref<?xf64>
    }
    %17 = memref.cast %16 : memref<?xf64> to memref<*xf64>
    func.call @read_input_2D_f64(%c0_i32, %c0, %c1, %9, %11, %13, %15, %17, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %18 = memref.alloc() : memref<7xindex>
    %19 = memref.cast %18 : memref<7xindex> to memref<*xindex>
    func.call @read_input_sizes_2D_f64(%c0_i32, %c0, %c1, %19, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, memref<*xindex>, i32) -> ()
    %20 = memref.load %18[%c0] : memref<7xindex>
    %21 = memref.load %18[%c1] : memref<7xindex>
    %22 = memref.load %18[%c2] : memref<7xindex>
    %23 = memref.load %18[%c3] : memref<7xindex>
    %24 = memref.load %18[%c4] : memref<7xindex>
    %25 = memref.load %18[%c6] : memref<7xindex>
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
    %30 = memref.alloc(%22) : memref<?xindex> // %30 = B.rowptr
    scf.for %arg0 = %c0 to %22 step %c1 {
      memref.store %c0, %30[%arg0] : memref<?xindex>
    }
    %31 = memref.cast %30 : memref<?xindex> to memref<*xindex>
    %32 = memref.alloc(%23) : memref<?xindex> // %32 = B.col
    scf.for %arg0 = %c0 to %23 step %c1 {
      memref.store %c0, %32[%arg0] : memref<?xindex>
    }
    %33 = memref.cast %32 : memref<?xindex> to memref<*xindex>
    %34 = memref.alloc(%24) : memref<?xf64> // %34 = B.val
    scf.for %arg0 = %c0 to %24 step %c1 {
      memref.store %cst, %34[%arg0] : memref<?xf64>
    }
    %35 = memref.cast %34 : memref<?xf64> to memref<*xf64>
    func.call @read_input_2D_f64(%c0_i32, %c0, %c1, %27, %29, %31, %33, %35, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %36 = arith.addi %7, %c1 : index
    // %37 = muli %7, %25 : index
    %37 = arith.constant 2000000000 : index

    %38 = memref.alloc(%c1) : memref<?xindex>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      memref.store %c0, %38[%arg0] : memref<?xindex>
    }
    %39 = memref.alloc(%c1) : memref<?xindex>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      memref.store %c0, %39[%arg0] : memref<?xindex>
    }
    %40 = memref.alloc(%36) : memref<?xindex> // %40 = C.rowptr
    scf.for %arg0 = %c0 to %36 step %c1 {
      memref.store %c0, %40[%arg0] : memref<?xindex>
    }
    %41 = memref.alloc(%37) : memref<?xindex> // %41 = C.col
    scf.for %arg0 = %c0 to %37 step %c1 {
      memref.store %c0, %41[%arg0] : memref<?xindex>
    }
    %42 = memref.alloc(%37) : memref<?xf64> // %42 = C.val
    scf.for %arg0 = %c0 to %37 step %c1 {
      memref.store %cst, %42[%arg0] : memref<?xf64>
    }
    %43 = memref.alloc() : memref<1xindex>
    memref.store %c1, %43[%c0] : memref<1xindex>
    %44 = memref.alloc() : memref<1xindex>
    memref.store %c1, %44[%c0] : memref<1xindex>
    %45 = memref.alloc() : memref<1xindex>
    memref.store %c1, %45[%c0] : memref<1xindex>
    %46 = memref.alloc() : memref<1xindex>
    memref.store %c0, %46[%c0] : memref<1xindex>
    %47 = memref.alloc() : memref<1xindex>
    memref.store %c0, %47[%c0] : memref<1xindex>
    memref.store %7, %38[%c0] : memref<?xindex>
    %48 = memref.alloc() : memref<7xindex>
    %49 = memref.cast %48 : memref<7xindex> to memref<*xindex>
    func.call @read_input_sizes_2D_f64(%c0_i32, %c0, %c1, %49, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, memref<*xindex>, i32) -> ()
    %50 = memref.load %48[%c0] : memref<7xindex>
    %51 = memref.load %48[%c1] : memref<7xindex>
    %52 = memref.load %48[%c2] : memref<7xindex>
    %53 = memref.load %48[%c3] : memref<7xindex>
    %54 = memref.load %48[%c4] : memref<7xindex>
    %55 = memref.alloc(%50) : memref<?xindex>
    scf.for %arg0 = %c0 to %50 step %c1 {
      memref.store %c0, %55[%arg0] : memref<?xindex>
    }
    %56 = memref.cast %55 : memref<?xindex> to memref<*xindex>
    %57 = memref.alloc(%51) : memref<?xindex>
    scf.for %arg0 = %c0 to %51 step %c1 {
      memref.store %c0, %57[%arg0] : memref<?xindex>
    }
    %58 = memref.cast %57 : memref<?xindex> to memref<*xindex>
    %59 = memref.alloc(%52) : memref<?xindex> // %59 = M.rowptr
    scf.for %arg0 = %c0 to %52 step %c1 {
      memref.store %c0, %59[%arg0] : memref<?xindex>
    }
    %60 = memref.cast %59 : memref<?xindex> to memref<*xindex>
    %61 = memref.alloc(%53) : memref<?xindex> // %61 = M.col
    scf.for %arg0 = %c0 to %53 step %c1 {
      memref.store %c0, %61[%arg0] : memref<?xindex>
    }
    %62 = memref.cast %61 : memref<?xindex> to memref<*xindex>
    %63 = memref.alloc(%54) : memref<?xf64> // %63 = M.val
    scf.for %arg0 = %c0 to %54 step %c1 {
      memref.store %cst, %63[%arg0] : memref<?xf64>
    }
    %64 = memref.cast %63 : memref<?xf64> to memref<*xf64>
    func.call @read_input_2D_f64(%c0_i32, %c0, %c1, %56, %58, %60, %62, %64, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()

    /// Changed D's type from f64 to i1, in order to use arith.cmpi
    %65 = memref.alloc(%25) {alignment = 32 : i64} : memref<?xi1> // %65 = D
    // %65 = memref.alloc(%25) {alignment = 32 : i64} : memref<?xf64> // %65 = D
    scf.for %arg0 = %c0 to %25 step %c1 {
      memref.store %false, %65[%arg0] : memref<?xi1>
      // memref.store %cst, %65[%arg0] : memref<?xf64>
    }

    %66 = func.call @getTime() : () -> f64

    %67 = memref.alloc(%25) {alignment = 32 : i64} : memref<?xf64> // %67 = workspace
    scf.for %arg0 = %c0 to %25 step %c1 {
      memref.store %cst, %67[%arg0] : memref<?xf64>
    }
    %68 = memref.alloc(%25) {alignment = 32 : i64} : memref<?xi1>
    scf.for %arg0 = %c0 to %25 step %c1 {
      memref.store %false, %68[%arg0] : memref<?xi1>
    }
    %69 = memref.alloc(%25) {alignment = 32 : i64} : memref<?xindex> // %69 = workspace.rowlist
    scf.for %arg0 = %c0 to %25 step %c1 {
      memref.store %c0, %69[%arg0] : memref<?xindex>
    }
    %70 = memref.alloc() : memref<1xindex>
    %71 = memref.load %8[%c0] : memref<?xindex>
    scf.for %arg0 = %c0 to %71 step %c1 {
      memref.store %c0, %70[%c0] : memref<1xindex>
      %78 = arith.addi %arg0, %c1 : index

      /// Initialize the temporary vector D as the dense masking vector
      %173 = memref.load %59[%arg0] : memref<?xindex> // %59 = M.rowptr
      %174 = memref.load %59[%78] : memref<?xindex> 
      scf.for %arg11 = %173 to %174 step %c1 {
        %176 = memref.load %63[%arg11] : memref<?xf64>  // %63 = M.val 
        %177 = arith.cmpf une, %176, %cst : f64
        scf.if %177 {
        // %177 = memref.cast %176 : f64 to index //  cast to index
        // %178 = arith.cmpi ne, %177, %c0 : index // if (%177 != 0)
        // scf.if %178 {
          %175 = memref.load %61[%arg11] : memref<?xindex> // %61 = M.col, %175 = M_col_id
          memref.store %true, %65[%175] : memref<?xi1> // %65 = D, D[M_col_id] = true;
        }
      }

      %79 = memref.load %12[%arg0] : memref<?xindex> // %12 = A.rowptr
      %80 = memref.load %12[%78] : memref<?xindex>
      scf.for %arg1 = %79 to %80 step %c1 {
        %86 = memref.load %14[%arg1] : memref<?xindex> // %14 = A.col, %86 = A_col_id
        %87 = arith.addi %86, %c1 : index
        %88 = memref.load %30[%86] : memref<?xindex> // %30 = B.rowptr
        %89 = memref.load %30[%87] : memref<?xindex>
        scf.for %arg2 = %88 to %89 step %c1 {
          %90 = memref.load %32[%arg2] : memref<?xindex> // %32 = B.col, %90 = B_col_id

          /// Filtering using dense vector D
          %179 = memref.load %65[%90] : memref<?xi1> // %65 = D, %179 = D[B_col_id]
          %180 = arith.cmpi eq, %179, %true : i1 // if (%179 == true) 
          scf.if %180 {

            %91 = memref.load %68[%90] : memref<?xi1>
            %92 = arith.cmpi eq, %91, %false : i1
            scf.if %92 {
              //%93 = memref.load %16[%arg1] : memref<?xf64>
              //%94 = memref.load %34[%arg2] : memref<?xf64>
              //%95 = arith.mulf %93, %94 : f64
              //memref.store %95, %67[%90] : memref<?xf64>
              memref.store %cst_1, %67[%90] : memref<?xf64>

              memref.store %true, %68[%90] : memref<?xi1>
              %96 = memref.load %70[%c0] : memref<1xindex>
              memref.store %90, %69[%96] : memref<?xindex>
              %97 = arith.addi %96, %c1 : index
              memref.store %97, %70[%c0] : memref<1xindex>
            } else {
              //%93 = memref.load %16[%arg1] : memref<?xf64>
              //%94 = memref.load %34[%arg2] : memref<?xf64>
              %95 = memref.load %67[%90] : memref<?xf64>
              //%96 = arith.mulf %93, %94 : f64
              //%97 = arith.addf %95, %96 : f64
              %97 = arith.addf %95, %cst_1 : f64

              memref.store %97, %67[%90] : memref<?xf64>
            }

          }
        }
      }

      /// Reset D
      scf.for %arg12 = %173 to %174 step %c1 {
        %175 = memref.load %61[%arg12] : memref<?xindex> // %61 = M.col, %175 = M_col_id
        memref.store %false, %65[%175] : memref<?xi1> // %65 = D, D[M_col_id] = true;
      }


      %81 = memref.load %70[%c0] : memref<1xindex>
      %82 = memref.cast %69 : memref<?xindex> to memref<*xindex>
      func.call @quick_sort(%82, %81) : (memref<*xindex>, index) -> ()
      scf.for %arg1 = %c0 to %81 step %c1 {
        %86 = memref.load %47[%c0] : memref<1xindex>
        %87 = memref.load %69[%arg1] : memref<?xindex>
        %88 = memref.load %67[%87] : memref<?xf64>
        memref.store %88, %42[%86] : memref<?xf64>
        memref.store %false, %68[%87] : memref<?xi1>
        memref.store %87, %41[%86] : memref<?xindex>
        %89 = arith.addi %86, %c1 : index
        memref.store %89, %47[%c0] : memref<1xindex>
        memref.store %89, %46[%c0] : memref<1xindex>
      }
      %83 = memref.load %45[%c0] : memref<1xindex>
      %84 = memref.load %46[%c0] : memref<1xindex>
      memref.store %84, %40[%83] : memref<?xindex>
      %85 = arith.addi %83, %c1 : index
      memref.store %85, %45[%c0] : memref<1xindex>
    }
    %887 = memref.alloc() : memref<1xindex>         // sum global
    memref.store %c0, %887[%c0] : memref<1xindex> // init sum
    %996 = memref.load %47[%c0] : memref<1xindex>   // loop limit
    scf.for %arg0 = %c0 to %996 step %c1 {       
      %889 = memref.load %42[%arg0] : memref<?xf64>    // get NNZs
      %890 = arith.fptoui %889 : f64 to i64
      %891 = arith.index_cast %890 : i64 to index 
      %900 = memref.load %887[%c0] : memref<1xindex>     // load sum global
      %911 = arith.addi %891, %900 : index                     // cur NNZ + sum global
      memref.store %911, %887[%c0] : memref<1xindex>     // update global sum
    }
    
    %72 = func.call @getTime() : () -> f64
    func.call @printElapsedTime(%66, %72) : (f64, f64) -> ()

    %888 = memref.cast %887 : memref<1xindex> to memref<*xindex>
    func.call @comet_print_memref_i64(%888) : (memref<*xindex>) -> ()

    // %73 = memref.cast %38 : memref<?xindex> to memref<*xindex>
    // func.call @comet_print_memref_i64(%73) : (memref<*xindex>) -> ()
    // %74 = memref.cast %39 : memref<?xindex> to memref<*xindex>
    // func.call @comet_print_memref_i64(%74) : (memref<*xindex>) -> ()
    // %75 = memref.cast %40 : memref<?xindex> to memref<*xindex>
    // func.call @comet_print_memref_i64(%75) : (memref<*xindex>) -> ()
    // %76 = memref.cast %41 : memref<?xindex> to memref<*xindex>
    // func.call @comet_print_memref_i64(%76) : (memref<*xindex>) -> ()
    // %77 = memref.cast %42 : memref<?xf64> to memref<*xf64>
    // func.call @comet_print_memref_f64(%77) : (memref<*xf64>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, memref<*xindex>, i32)
  func.func private @quick_sort(memref<*xindex>, index)
  func.func private @getTime() -> f64
  func.func private @printElapsedTime(f64, f64)
  func.func private @comet_print_memref_f64(memref<*xf64>)
  func.func private @comet_print_memref_i64(memref<*xindex>)
}
