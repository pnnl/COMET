module {
  func.func @main() {
    %cst = arith.constant 1.700000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<13xindex>
    %cast = memref.cast %alloc : memref<13xindex> to memref<*xindex>
    call @read_input_sizes_2D_f64(%c0_i32, %c0, %c0, %c1, %c0, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
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
    call @read_input_2D_f64(%c0_i32, %c0, %c0, %c1, %c0, %cast_2, %cast_4, %cast_6, %cast_8, %cast_10, %cast_12, %cast_14, %cast_16, %cast_18, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %alloc_19 = memref.alloc(%10) {alignment = 32 : i64} : memref<?xf64>
    %alloc_20 = memref.alloc(%9) {alignment = 32 : i64} : memref<?xf64>
    linalg.fill ins(%cst : f64) outs(%alloc_19 : memref<?xf64>)
    linalg.fill ins(%cst_0 : f64) outs(%alloc_20 : memref<?xf64>)
    
    //; n1 -> Range(A1_pos)
    %11 = memref.load %alloc_1[%c0] : memref<?xindex>
    scf.for %arg0 = %c0 to %11 step %c1 {
      
      //; bi -> range(A1_block_pos)
      %12 = memref.load %alloc_5[%c0] : memref<?xindex>
      scf.for %arg1 = %c0 to %12 step %c1 {
        
        //; A1_block_pos
        %13 = memref.load %alloc_5[%c0] : memref<?xindex>
        //; = n1 * A1_block_pos
        %14 = arith.muli %arg0, %13 : index
        //; = (n1 * A1_block_pos) + bi
        %15 = arith.addi %14, %arg1 : index
        
        //; Rest of the equation- good (CU portion)
        %16 = arith.addi %arg0, %c1 : index
        %17 = memref.load %alloc_9[%arg0] : memref<?xindex>
        %18 = memref.load %alloc_9[%16] : memref<?xindex>
        
        // ;n2 (CU)
        scf.for %arg2 = %17 to %18 step %c1 {
          //; A2crd[n2]
          %19 = memref.load %alloc_11[%arg2] : memref<?xindex>
          //; A2_block_pos
          %20 = memref.load %alloc_13[%c0] : memref<?xindex>
          
          //; bj -> range(A2_block_pos)
          scf.for %arg3 = %c0 to %20 step %c1 {
            //; A2_block_pos
            %21 = memref.load %alloc_13[%c1] : memref<?xindex>
            //; A2crd[n2] * A2_block_pos
            %22 = arith.muli %19, %21 : index
            //; (A2crd[n2] + A2_block_pos) + bj
            %23 = arith.addi %22, %arg3 : index
            
            //; A1_block_pos
            %24 = memref.load %alloc_5[%c0] : memref<?xindex>
            //; A2_block_pos
            %25 = memref.load %alloc_13[%c0] : memref<?xindex>
            //; = %26 = (A1_block_pos * A2_block_pos)
            %26 = arith.muli %24, %25 : index
            // ; %27 = n2 * %26 (CHANGED)
            %27 = arith.muli %arg2, %26 : index
            //; * bi (CHANGED)
            %28 = arith.muli %arg1, %25 : index
            %29 = arith.addi %27, %28 : index
            %30 = arith.addi %29, %arg3 : index
            %31 = memref.load %alloc_17[%30] : memref<?xf64>
            %32 = memref.load %alloc_19[%23] : memref<?xf64>
            %33 = memref.load %alloc_20[%15] : memref<?xf64>
            %34 = arith.mulf %31, %32 : f64
            %35 = arith.addf %33, %34 : f64
            memref.store %35, %alloc_20[%15] : memref<?xf64>
          }
        }
      }
    }
    %cast_21 = memref.cast %alloc_20 : memref<?xf64> to memref<*xf64>
    call @comet_print_memref_f64(%cast_21) : (memref<*xf64>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @comet_sort_index(memref<*xindex>, index, index)
  func.func private @comet_print_memref_f64(memref<*xf64>)
}
