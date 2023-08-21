// ../llvm/build/bin/mlir-opt bfs.SCF.v5.yield_for_loop.updated.mlir | ../llvm/build/bin/mlir-opt --convert-linalg-to-loops --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts &> bfs.llvm
// ../llvm/build/bin/mlir-cpu-runner bfs.llvm -O3 -e main -entry-point-result=void -shared-libs=./lib/libcomet_runner_utils.dylib
module  {
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant 1.000000e+00 : f64
    %c1717 = arith.constant 1717 : index
    %true = arith.constant true
    %false = arith.constant false

    %bi_mem = memref.alloc() : memref<1xindex>
    %bi = memref.cast %bi_mem : memref<1xindex> to memref<*xindex>

    %0 = memref.alloc() : memref<7xindex>
    %1 = memref.cast %0 : memref<7xindex> to memref<*xindex>
    call @read_input_sizes_2D_f64(%c0_i32, %c0, %c1, %1, %c0_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, memref<*xindex>, i32) -> ()
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
    %13 = memref.alloc(%4) : memref<?xindex> // %13 = A.rowptr
    scf.for %arg0 = %c0 to %4 step %c1 {
      memref.store %c0, %13[%arg0] : memref<?xindex>
    }
    %14 = memref.cast %13 : memref<?xindex> to memref<*xindex>
    %15 = memref.alloc(%5) : memref<?xindex> // %15 = A.col
    scf.for %arg0 = %c0 to %5 step %c1 {
      memref.store %c0, %15[%arg0] : memref<?xindex>
    }
    %16 = memref.cast %15 : memref<?xindex> to memref<*xindex>
    %17 = memref.alloc(%6) : memref<?xf64> // %16 = A.val
    scf.for %arg0 = %c0 to %6 step %c1 {
      memref.store %cst_0, %17[%arg0] : memref<?xf64>
    }
    %18 = memref.cast %17 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c0_i32, %c0, %c1, %10, %12, %14, %16, %18, %c0_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()

    %frt_size = memref.alloc() : memref<1xindex> // fronter.size
    %frt_col = memref.alloc(%7) {alignment = 32 : i64} : memref<?xindex> // frontier.col
    // %frt_val = memref.alloc(%7) {alignment = 32 : i64} : memref<?xi1> // frontier.val

    // memref.store %c0, %frt_col[%c0] : memref<?xindex> // frontier.col[0] = 0
    memref.store %c1717, %frt_col[%c0] : memref<?xindex> // frontier.col[0] = 1717
    memref.store %c1, %frt_size[%c0] : memref<1xindex> // frontier.size = 1

    %l_level = memref.alloc(%8) {alignment = 32 : i64} : memref<?xindex>  // l_level
    scf.for %arg0 = %c0 to %8 step %c1 {               // l_level
      memref.store %c0, %l_level[%arg0] : memref<?xindex>
    }
    // memref.store %c1, %l_level[%c0] : memref<?xindex> // l_level[0] = 1


    /// Workspace
    %workspace = memref.alloc(%8) {alignment = 32 : i64} : memref<?xi1> // workspace
    scf.for %arg0 = %c0 to %8 step %c1 {
      memref.store %false, %workspace[%arg0] : memref<?xi1>
    }


    // /// Count
    // %count = memref.alloc() : memref<1xindex>
    // memref.store %c1, %count[%c0] : memref<1xindex>

    // timing start
    %time_start = call @getTime() : () -> f64

    //scf.for %arg0 = %c1 to %c2_N step %c1 {    // repeat loop
    %level_bound = arith.addi %8, %c1 : index
    scf.for %arg0 = %c1 to %level_bound step %c1 {    // repeat loop

        // /// Test
        // %21 = memref.cast %l_level : memref<?xindex> to memref<*xindex>
        // call @comet_print_memref_i64(%21) : (memref<*xindex>) -> ()
        // /// End test
      
      %9027 = memref.load %frt_size[%c0] : memref<1xindex>
      // /// Check if finished
      // %is_zero = cmpi eq, %9027, %c0 : index
      // scf.if %is_zero {
      //   /// The front size is zero (i.e., finished)
      //   // memref.store %true, %is_finished[%c0] : memref<1xi1>
      //   scf.yield
      // }

      %is_not_zero = arith.cmpi ne, %9027, %c0 : index
      scf.if %is_not_zero {


      /// Update the l_level
      // %9027 = memref.load %frt_size[%c0] : memref<1xindex>
      scf.for %row_i = %c0 to %9027 step %c1 {
        %9028 = memref.load %frt_col[%row_i] : memref<?xindex>
        memref.store %arg0, %l_level[%9028] : memref<?xindex>
      }


      /// Do (frontier vxm(LOR.LAND) A), which is very like SpVM
      %row_bound = memref.load %frt_size[%c0] : memref<1xindex>
      scf.for %row_i = %c0 to %row_bound step %c1 {
        %f_col_id = memref.load %frt_col[%row_i] : memref<?xindex> // every f_col_id in frontier[:]
        %9020 = arith.addi %f_col_id, %c1 : index
        %A_row_start = memref.load %13[%f_col_id] : memref<?xindex> // %13 = A.rowptr
        %A_row_bound = memref.load %13[%9020] : memref<?xindex> 
        scf.for %col_i = %A_row_start to %A_row_bound step %c1 { // every A_col_id in A[f_col_id][:]
          %A_col_id = memref.load %15[%col_i] : memref<?xindex> // %15 = A.col

            // /// print f_col_id, A_col_id
            // %ibuffer = memref.alloc() : memref<1xindex>
            // %ivalue = memref.cast %ibuffer : memref<1xindex> to memref<*xindex>

            // memref.store %f_col_id, %ibuffer[%c0] : memref<1xindex>
            // call @comet_print_memref_i64(%ivalue) : (memref<*xindex>) -> ()

            // memref.store %A_col_id, %ibuffer[%c0] : memref<1xindex>
            // call @comet_print_memref_i64(%ivalue) : (memref<*xindex>) -> ()
            // /// End print f_col_id, A_col_id
          

          %9025 = memref.load %l_level[%A_col_id] : memref<?xindex> 
          %9026 = arith.cmpi eq, %9025, %c0 : index
          scf.if %9026 {  // if l_level[A_col_id] == 0
            memref.store %true, %workspace[%A_col_id] : memref<?xi1> // Mark in workspace
          }
        }
      }

      /// Update the frontier from the workspace
      memref.store %c0, %frt_size[%c0] : memref<1xindex> // frontier.size = 0
      scf.for %row_i = %c0 to %7 step %c1 {
        %9021 = memref.load %workspace[%row_i] : memref<?xi1>
        %9022 = arith.cmpi eq, %9021, %true : i1
        scf.if %9022 {  // if workspace[row_i] == true
          %9023 = memref.load %frt_size[%c0] : memref<1xindex>
          memref.store %row_i, %frt_col[%9023] : memref<?xindex> // frontier.col[size] = row_i
          %9024 = arith.addi %9023, %c1 : index // size++
          memref.store %9024, %frt_size[%c0] : memref<1xindex>
          memref.store %false, %workspace[%row_i] : memref<?xi1> // reset workspace[row_i] = false

            // /// print the row_i
            // %ibuffer = memref.alloc() : memref<1xindex>
            // %ivalue = memref.cast %ibuffer : memref<1xindex> to memref<*xindex>

            // memref.store %row_i, %ibuffer[%c0] : memref<1xindex>
            // call @comet_print_memref_i64(%ivalue) : (memref<*xindex>) -> ()
            // /// End print the row_i
        }
      }

      // %21 = memref.cast %l_level : memref<?xindex> to memref<*xindex>
      // call @comet_print_memref_i64(%21) : (memref<*xindex>) -> ()

      // %count_tmp = memref.load %count[%c0] : memref<1xindex>
      // %count_new = addi %count_tmp, %c1 : index
      // memref.store %count_new, %count[%c0] : memref<1xindex>

      // scf.yield %arg0 : index
      } // End if not zero
    }
    // timing end
    %time_end = call @getTime() : () -> f64
    call @printElapsedTime(%time_start, %time_end) : (f64, f64) -> ()

    // %21 = memref.cast %l_level : memref<?xindex> to memref<*xindex>
    // call @comet_print_memref_i64(%21) : (memref<*xindex>) -> ()

    // // Count
    // %count_val = memref.cast %count : memref<1xindex> to memref<*xindex>
    // call @comet_print_memref_i64(%count_val) : (memref<*xindex>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, memref<*xindex>, i32)
  func.func private @quick_sort(memref<*xindex>, index)
  func.func private @comet_print_memref_f64(memref<*xf64>)
  func.func private @comet_print_memref_i64(memref<*xindex>)
  func.func private @getTime() -> f64
  func.func private @printElapsedTime(f64, f64)
}
