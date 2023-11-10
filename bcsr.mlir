module {
  func.func @main() {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %alloc = memref.alloc() : memref<13xindex>
    %cast = memref.cast %alloc : memref<13xindex> to memref<*xindex>
    call @read_input_sizes_2D_f64(%c0_i32, %c0, %c0, %c2, %c0, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
    %0 = memref.load %alloc[%c0] : memref<13xindex>
    %1 = memref.load %alloc[%c1] : memref<13xindex>
    %2 = memref.load %alloc[%c2] : memref<13xindex>
    %3 = memref.load %alloc[%c3] : memref<13xindex>
    %4 = memref.load %alloc[%c4] : memref<13xindex>
    %5 = memref.load %alloc[%c5] : memref<13xindex>
    %6 = memref.load %alloc[%c6] : memref<13xindex>
    %7 = memref.load %alloc[%c7] : memref<13xindex>
    %8 = memref.load %alloc[%c8] : memref<13xindex>
    %9 = memref.load %alloc[%c9] : memref<13xindex>
    %10 = memref.load %alloc[%c10] : memref<13xindex>
    %alloc_1 = memref.alloc(%0) : memref<?xindex>
    scf.for %arg0 = %c0 to %0 step %c1 {
      memref.store %c0, %alloc_1[%arg0] : memref<?xindex>
    }
    %cast_2 = memref.cast %alloc_1 : memref<?xindex> to memref<*xindex>
    %alloc_3 = memref.alloc(%1) : memref<?xindex>
    scf.for %arg0 = %c0 to %1 step %c1 {
      memref.store %c0, %alloc_3[%arg0] : memref<?xindex>
    }
    %cast_4 = memref.cast %alloc_3 : memref<?xindex> to memref<*xindex>
    %alloc_5 = memref.alloc(%2) : memref<?xindex>
    scf.for %arg0 = %c0 to %2 step %c1 {
      memref.store %c0, %alloc_5[%arg0] : memref<?xindex>
    }
    %cast_6 = memref.cast %alloc_5 : memref<?xindex> to memref<*xindex>
    %alloc_7 = memref.alloc(%3) : memref<?xindex>
    scf.for %arg0 = %c0 to %3 step %c1 {
      memref.store %c0, %alloc_7[%arg0] : memref<?xindex>
    }
    %cast_8 = memref.cast %alloc_7 : memref<?xindex> to memref<*xindex>
    %alloc_9 = memref.alloc(%4) : memref<?xindex>
    scf.for %arg0 = %c0 to %4 step %c1 {
      memref.store %c0, %alloc_9[%arg0] : memref<?xindex>
    }
    %cast_10 = memref.cast %alloc_9 : memref<?xindex> to memref<*xindex>
    %alloc_11 = memref.alloc(%5) : memref<?xindex>
    scf.for %arg0 = %c0 to %5 step %c1 {
      memref.store %c0, %alloc_11[%arg0] : memref<?xindex>
    }
    %cast_12 = memref.cast %alloc_11 : memref<?xindex> to memref<*xindex>
    %alloc_13 = memref.alloc(%6) : memref<?xindex>
    scf.for %arg0 = %c0 to %6 step %c1 {
      memref.store %c0, %alloc_13[%arg0] : memref<?xindex>
    }
    %cast_14 = memref.cast %alloc_13 : memref<?xindex> to memref<*xindex>
    %alloc_15 = memref.alloc(%7) : memref<?xindex>
    scf.for %arg0 = %c0 to %7 step %c1 {
      memref.store %c0, %alloc_15[%arg0] : memref<?xindex>
    }
    %cast_16 = memref.cast %alloc_15 : memref<?xindex> to memref<*xindex>
    %alloc_17 = memref.alloc(%8) : memref<?xf64>
    scf.for %arg0 = %c0 to %8 step %c1 {
      memref.store %cst_0, %alloc_17[%arg0] : memref<?xf64>
    }
    %cast_18 = memref.cast %alloc_17 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c0_i32, %c0, %c0, %c2, %c0, %cast_2, %cast_4, %cast_6, %cast_8, %cast_10, %cast_12, %cast_14, %cast_16, %cast_18, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %alloc_19 = memref.alloc(%9) {alignment = 32 : i64} : memref<4x?xf64>
    %alloc_20 = memref.alloc(%10) {alignment = 32 : i64} : memref<4x?xf64>
    linalg.fill ins(%cst : f64) outs(%alloc_19 : memref<4x?xf64>)
    linalg.fill ins(%cst_0 : f64) outs(%alloc_20 : memref<4x?xf64>)

    //;scf.for %arg0 = %c0 to %c4 step %c1 {
    //;  scf.for %arg1 = %c0 to %9 step %c1 {
    //;    %11 = memref.load %alloc_9[%c0] : memref<?xindex>
    //;    %12 = memref.load %alloc_9[%c1] : memref<?xindex>
    //;    scf.for %arg2 = %11 to %12 step %c1 {
    //;      %13 = memref.load %alloc_11[%arg2] : memref<?xindex>
    //;      %14 = memref.load %alloc_19[%arg0, %arg1] : memref<4x?xf64>
    //;      %15 = memref.load %alloc_17[%arg2] : memref<?xf64>
    //;      %16 = memref.load %alloc_20[%arg0, %13] : memref<4x?xf64>
    //;      %17 = arith.mulf %14, %15 : f64
    //;      %18 = arith.addf %16, %17 : f64
    //;      memref.store %18, %alloc_20[%arg0, %13] : memref<4x?xf64>
    //;    }
    //;  }
    //;}
    
    %c_A1_pos = memref.load %alloc_1[%c0] : memref<?xindex>
    %c_A1_block = memref.load %alloc_5[%c0] : memref<?xindex>
    %c_A2_block = memref.load %alloc_13[%c0] : memref<?xindex>
    
    scf.for %n1 = %c0 to %c_A1_pos step %c1 {
        scf.for %bi = %c0 to %c_A1_block step %c1 {
            %n1_next = arith.addi %n1, %c1 : index
            %11 = memref.load %alloc_9[%n1] : memref<?xindex>
            %12 = memref.load %alloc_9[%n1_next] : memref<?xindex>
            scf.for %n2 = %11 to %12 step %c1 {
                scf.for %bj = %c0 to %c_A2_block step %c1 {
                    %i1 = arith.muli %n1, %c_A1_block : index
                    %i = arith.addi %i1, %bi : index 
                    
                    %13 = memref.load %alloc_11[%n2] : memref<?xindex>
                    %j1 = arith.muli %13, %c_A2_block : index
                    %j = arith.addi %j1, %bj : index
                    
                    %p1 = arith.muli %c_A1_block, %c_A2_block : index
                    %p2 = arith.muli %p1, %n2 : index
                    %p3 = arith.muli %bi, %c_A2_block : index
                    %p4 = arith.addi %p2, %p3 : index
                    %p = arith.addi %p4, %bj : index
                    
                    scf.for %k = %c0 to %c6 step %c1 {
                        %14 = memref.load %alloc_19[%j, %k] : memref<4x?xf64>
                        %15 = memref.load %alloc_17[%p] : memref<?xf64>
                        %16 = memref.load %alloc_20[%i, %k] : memref<4x?xf64>
                        %17 = arith.mulf %14, %15 : f64
                        %18 = arith.addf %16, %17 : f64
                        memref.store %18, %alloc_20[%i, %k] : memref<4x?xf64>
                    }
                }
            }
        }
    }
    
    %cast_21 = memref.cast %alloc_19 : memref<4x?xf64> to memref<*xf64>
    call @comet_print_memref_f64(%cast_21) : (memref<*xf64>) -> ()
    call @comet_print_memref_i64(%cast_2) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_4) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_6) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_8) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_10) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_12) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_14) : (memref<*xindex>) -> ()
    call @comet_print_memref_i64(%cast_16) : (memref<*xindex>) -> ()
    call @comet_print_memref_f64(%cast_18) : (memref<*xf64>) -> ()
    %cast_22 = memref.cast %alloc_20 : memref<4x?xf64> to memref<*xf64>
    call @comet_print_memref_f64(%cast_22) : (memref<*xf64>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @comet_sort_index(memref<*xindex>, index, index)
  func.func private @comet_print_memref_f64(memref<*xf64>)
  func.func private @comet_print_memref_i64(memref<*xindex>)
}
