/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4560 LowerIndexTreeToSCFPass
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4574 %36 = "it.itree"(%35) : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4277 
doLoweringIndexTreeToSCF in LowerIndexTreeIRToSCF
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4281 module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = "ta.static_index_label"(%c0, %c4, %c1) : (index, index, index) -> !ta.range
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %2 = "ta.dynamic_index_label"(%c0_2, %c1_3) : (index, index) -> !ta.range
    %alloc = memref.alloc() : memref<13xindex>
    %cast = memref.cast %alloc : memref<13xindex> to memref<*xindex>
    %c-1 = arith.constant -1 : index
    %c0_4 = arith.constant 0 : index
    %c1_5 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    call @read_input_sizes_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
    %c0_6 = arith.constant 0 : index
    %3 = memref.load %alloc[%c0_6] : memref<13xindex>
    %c1_7 = arith.constant 1 : index
    %4 = memref.load %alloc[%c1_7] : memref<13xindex>
    %c2_8 = arith.constant 2 : index
    %5 = memref.load %alloc[%c2_8] : memref<13xindex>
    %c3_9 = arith.constant 3 : index
    %6 = memref.load %alloc[%c3_9] : memref<13xindex>
    %c4_10 = arith.constant 4 : index
    %7 = memref.load %alloc[%c4_10] : memref<13xindex>
    %c5 = arith.constant 5 : index
    %8 = memref.load %alloc[%c5] : memref<13xindex>
    %c6 = arith.constant 6 : index
    %9 = memref.load %alloc[%c6] : memref<13xindex>
    %c7 = arith.constant 7 : index
    %10 = memref.load %alloc[%c7] : memref<13xindex>
    %c8 = arith.constant 8 : index
    %11 = memref.load %alloc[%c8] : memref<13xindex>
    %c9 = arith.constant 9 : index
    %12 = memref.load %alloc[%c9] : memref<13xindex>
    %c10 = arith.constant 10 : index
    %13 = memref.load %alloc[%c10] : memref<13xindex>
    %c11 = arith.constant 11 : index
    %14 = memref.load %alloc[%c11] : memref<13xindex>
    %c12 = arith.constant 12 : index
    %15 = memref.load %alloc[%c12] : memref<13xindex>
    %alloc_11 = memref.alloc(%3) : memref<?xindex>
    %c0_12 = arith.constant 0 : index
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    scf.for %arg0 = %c0_13 to %3 step %c1_14 {
      memref.store %c0_12, %alloc_11[%arg0] : memref<?xindex>
    }
    %cast_15 = memref.cast %alloc_11 : memref<?xindex> to memref<*xindex>
    %alloc_16 = memref.alloc(%4) : memref<?xindex>
    %c0_17 = arith.constant 0 : index
    %c0_18 = arith.constant 0 : index
    %c1_19 = arith.constant 1 : index
    scf.for %arg0 = %c0_18 to %4 step %c1_19 {
      memref.store %c0_17, %alloc_16[%arg0] : memref<?xindex>
    }
    %cast_20 = memref.cast %alloc_16 : memref<?xindex> to memref<*xindex>
    %alloc_21 = memref.alloc(%5) : memref<?xindex>
    %c0_22 = arith.constant 0 : index
    %c0_23 = arith.constant 0 : index
    %c1_24 = arith.constant 1 : index
    scf.for %arg0 = %c0_23 to %5 step %c1_24 {
      memref.store %c0_22, %alloc_21[%arg0] : memref<?xindex>
    }
    %cast_25 = memref.cast %alloc_21 : memref<?xindex> to memref<*xindex>
    %alloc_26 = memref.alloc(%6) : memref<?xindex>
    %c0_27 = arith.constant 0 : index
    %c0_28 = arith.constant 0 : index
    %c1_29 = arith.constant 1 : index
    scf.for %arg0 = %c0_28 to %6 step %c1_29 {
      memref.store %c0_27, %alloc_26[%arg0] : memref<?xindex>
    }
    %cast_30 = memref.cast %alloc_26 : memref<?xindex> to memref<*xindex>
    %alloc_31 = memref.alloc(%7) : memref<?xindex>
    %c0_32 = arith.constant 0 : index
    %c0_33 = arith.constant 0 : index
    %c1_34 = arith.constant 1 : index
    scf.for %arg0 = %c0_33 to %7 step %c1_34 {
      memref.store %c0_32, %alloc_31[%arg0] : memref<?xindex>
    }
    %cast_35 = memref.cast %alloc_31 : memref<?xindex> to memref<*xindex>
    %alloc_36 = memref.alloc(%8) : memref<?xindex>
    %c0_37 = arith.constant 0 : index
    %c0_38 = arith.constant 0 : index
    %c1_39 = arith.constant 1 : index
    scf.for %arg0 = %c0_38 to %8 step %c1_39 {
      memref.store %c0_37, %alloc_36[%arg0] : memref<?xindex>
    }
    %cast_40 = memref.cast %alloc_36 : memref<?xindex> to memref<*xindex>
    %alloc_41 = memref.alloc(%9) : memref<?xindex>
    %c0_42 = arith.constant 0 : index
    %c0_43 = arith.constant 0 : index
    %c1_44 = arith.constant 1 : index
    scf.for %arg0 = %c0_43 to %9 step %c1_44 {
      memref.store %c0_42, %alloc_41[%arg0] : memref<?xindex>
    }
    %cast_45 = memref.cast %alloc_41 : memref<?xindex> to memref<*xindex>
    %alloc_46 = memref.alloc(%10) : memref<?xindex>
    %c0_47 = arith.constant 0 : index
    %c0_48 = arith.constant 0 : index
    %c1_49 = arith.constant 1 : index
    scf.for %arg0 = %c0_48 to %10 step %c1_49 {
      memref.store %c0_47, %alloc_46[%arg0] : memref<?xindex>
    }
    %cast_50 = memref.cast %alloc_46 : memref<?xindex> to memref<*xindex>
    %alloc_51 = memref.alloc(%11) : memref<?xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %c0_52 = arith.constant 0 : index
    %c1_53 = arith.constant 1 : index
    scf.for %arg0 = %c0_52 to %11 step %c1_53 {
      memref.store %cst, %alloc_51[%arg0] : memref<?xf64>
    }
    %cast_54 = memref.cast %alloc_51 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast_15, %cast_20, %cast_25, %cast_30, %cast_35, %cast_40, %cast_45, %cast_50, %cast_54, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %16 = bufferization.to_tensor %alloc_11 : memref<?xindex>
    %17 = bufferization.to_tensor %alloc_16 : memref<?xindex>
    %18 = bufferization.to_tensor %alloc_21 : memref<?xindex>
    %19 = bufferization.to_tensor %alloc_26 : memref<?xindex>
    %20 = bufferization.to_tensor %alloc_31 : memref<?xindex>
    %21 = bufferization.to_tensor %alloc_36 : memref<?xindex>
    %22 = bufferization.to_tensor %alloc_41 : memref<?xindex>
    %23 = bufferization.to_tensor %alloc_46 : memref<?xindex>
    %24 = bufferization.to_tensor %alloc_51 : memref<?xf64>
    %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
    %26 = "ta.static_index_label"(%c0_0, %12, %c1_1) : (index, index, index) -> !ta.range
    %27 = "ta.static_index_label"(%c0_2, %13, %c1_3) : (index, index, index) -> !ta.range
    %alloc_55 = memref.alloc(%12) {alignment = 32 : i64} : memref<4x?xf64>
    %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
    %alloc_56 = memref.alloc(%13) {alignment = 32 : i64} : memref<4x?xf64>
    %29 = bufferization.to_tensor %alloc_56 : memref<4x?xf64>
    %cst_57 = arith.constant 1.000000e+00 : f64
    linalg.fill ins(%cst_57 : f64) outs(%alloc_55 : memref<4x?xf64>)
    %cst_58 = arith.constant 0.000000e+00 : f64
    linalg.fill ins(%cst_58 : f64) outs(%alloc_56 : memref<4x?xf64>)
    %30 = "it.ComputeRHS"(%28, %25) {allBlocks = [["UNK", "UNK"], ["D", "D"]], allFormats = [["D", "D"], ["D", "CN"]], allPerms = [[0, 1], [1, 2]]} : (tensor<4x?xf64>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> tensor<*xf64>
    %31 = "it.ComputeLHS"(%29) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[0, 2]]} : (tensor<4x?xf64>) -> tensor<*xf64>
    %32 = "it.Compute"(%30, %31) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
    %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
    %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
    %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
    %36 = "it.itree"(%35) : (i64) -> i64
    "ta.print"(%29) : (tensor<4x?xf64>) -> ()
    "ta.print"(%25) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @comet_sort_index(memref<*xindex>, index, index)
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4314  0 /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4315 %36 = "it.itree"(%35) : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4318  parent: 4
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4314  1 /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4315 %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4318  parent: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4314  2 /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4315 %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4318  parent: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4314  3 /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4315 %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4318  parent: 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4350  
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4350  
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4350  
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4350  
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4394  i: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4395 %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4400 curOp is IndexTreeIndicesOp
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4401 %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4411  indices.size(): 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4442 %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4452  indices.size(): 1 tensors.size(): 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4455  Formats:D 0 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4456  Blocks:UNK 0 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4457 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4458 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4460 ---------------
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4470  call genForOps, i = 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1053  genForOps indexTreeOp
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1054 %36 = "it.itree"(%35) : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1059  genForOps ancestorsOps.size(): 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1066 Tensor size: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:492 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:496  getAllocs() -  it is dense
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:501 %alloc_55 = memref.alloc(%12) {alignment = 32 : i64} : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1069 Tensors:
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1072 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1096  current index format: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1097  current index block: UNK
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:640  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:641 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:680  D Loop
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:681 scf.for %arg0 = %c0_60 to %c4_59 step %c1_61 {
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1286 finish generate loops for current index format: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4473 module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = "ta.static_index_label"(%c0, %c4, %c1) : (index, index, index) -> !ta.range
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %2 = "ta.dynamic_index_label"(%c0_2, %c1_3) : (index, index) -> !ta.range
    %alloc = memref.alloc() : memref<13xindex>
    %cast = memref.cast %alloc : memref<13xindex> to memref<*xindex>
    %c-1 = arith.constant -1 : index
    %c0_4 = arith.constant 0 : index
    %c1_5 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    call @read_input_sizes_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
    %c0_6 = arith.constant 0 : index
    %3 = memref.load %alloc[%c0_6] : memref<13xindex>
    %c1_7 = arith.constant 1 : index
    %4 = memref.load %alloc[%c1_7] : memref<13xindex>
    %c2_8 = arith.constant 2 : index
    %5 = memref.load %alloc[%c2_8] : memref<13xindex>
    %c3_9 = arith.constant 3 : index
    %6 = memref.load %alloc[%c3_9] : memref<13xindex>
    %c4_10 = arith.constant 4 : index
    %7 = memref.load %alloc[%c4_10] : memref<13xindex>
    %c5 = arith.constant 5 : index
    %8 = memref.load %alloc[%c5] : memref<13xindex>
    %c6 = arith.constant 6 : index
    %9 = memref.load %alloc[%c6] : memref<13xindex>
    %c7 = arith.constant 7 : index
    %10 = memref.load %alloc[%c7] : memref<13xindex>
    %c8 = arith.constant 8 : index
    %11 = memref.load %alloc[%c8] : memref<13xindex>
    %c9 = arith.constant 9 : index
    %12 = memref.load %alloc[%c9] : memref<13xindex>
    %c10 = arith.constant 10 : index
    %13 = memref.load %alloc[%c10] : memref<13xindex>
    %c11 = arith.constant 11 : index
    %14 = memref.load %alloc[%c11] : memref<13xindex>
    %c12 = arith.constant 12 : index
    %15 = memref.load %alloc[%c12] : memref<13xindex>
    %alloc_11 = memref.alloc(%3) : memref<?xindex>
    %c0_12 = arith.constant 0 : index
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    scf.for %arg0 = %c0_13 to %3 step %c1_14 {
      memref.store %c0_12, %alloc_11[%arg0] : memref<?xindex>
    }
    %cast_15 = memref.cast %alloc_11 : memref<?xindex> to memref<*xindex>
    %alloc_16 = memref.alloc(%4) : memref<?xindex>
    %c0_17 = arith.constant 0 : index
    %c0_18 = arith.constant 0 : index
    %c1_19 = arith.constant 1 : index
    scf.for %arg0 = %c0_18 to %4 step %c1_19 {
      memref.store %c0_17, %alloc_16[%arg0] : memref<?xindex>
    }
    %cast_20 = memref.cast %alloc_16 : memref<?xindex> to memref<*xindex>
    %alloc_21 = memref.alloc(%5) : memref<?xindex>
    %c0_22 = arith.constant 0 : index
    %c0_23 = arith.constant 0 : index
    %c1_24 = arith.constant 1 : index
    scf.for %arg0 = %c0_23 to %5 step %c1_24 {
      memref.store %c0_22, %alloc_21[%arg0] : memref<?xindex>
    }
    %cast_25 = memref.cast %alloc_21 : memref<?xindex> to memref<*xindex>
    %alloc_26 = memref.alloc(%6) : memref<?xindex>
    %c0_27 = arith.constant 0 : index
    %c0_28 = arith.constant 0 : index
    %c1_29 = arith.constant 1 : index
    scf.for %arg0 = %c0_28 to %6 step %c1_29 {
      memref.store %c0_27, %alloc_26[%arg0] : memref<?xindex>
    }
    %cast_30 = memref.cast %alloc_26 : memref<?xindex> to memref<*xindex>
    %alloc_31 = memref.alloc(%7) : memref<?xindex>
    %c0_32 = arith.constant 0 : index
    %c0_33 = arith.constant 0 : index
    %c1_34 = arith.constant 1 : index
    scf.for %arg0 = %c0_33 to %7 step %c1_34 {
      memref.store %c0_32, %alloc_31[%arg0] : memref<?xindex>
    }
    %cast_35 = memref.cast %alloc_31 : memref<?xindex> to memref<*xindex>
    %alloc_36 = memref.alloc(%8) : memref<?xindex>
    %c0_37 = arith.constant 0 : index
    %c0_38 = arith.constant 0 : index
    %c1_39 = arith.constant 1 : index
    scf.for %arg0 = %c0_38 to %8 step %c1_39 {
      memref.store %c0_37, %alloc_36[%arg0] : memref<?xindex>
    }
    %cast_40 = memref.cast %alloc_36 : memref<?xindex> to memref<*xindex>
    %alloc_41 = memref.alloc(%9) : memref<?xindex>
    %c0_42 = arith.constant 0 : index
    %c0_43 = arith.constant 0 : index
    %c1_44 = arith.constant 1 : index
    scf.for %arg0 = %c0_43 to %9 step %c1_44 {
      memref.store %c0_42, %alloc_41[%arg0] : memref<?xindex>
    }
    %cast_45 = memref.cast %alloc_41 : memref<?xindex> to memref<*xindex>
    %alloc_46 = memref.alloc(%10) : memref<?xindex>
    %c0_47 = arith.constant 0 : index
    %c0_48 = arith.constant 0 : index
    %c1_49 = arith.constant 1 : index
    scf.for %arg0 = %c0_48 to %10 step %c1_49 {
      memref.store %c0_47, %alloc_46[%arg0] : memref<?xindex>
    }
    %cast_50 = memref.cast %alloc_46 : memref<?xindex> to memref<*xindex>
    %alloc_51 = memref.alloc(%11) : memref<?xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %c0_52 = arith.constant 0 : index
    %c1_53 = arith.constant 1 : index
    scf.for %arg0 = %c0_52 to %11 step %c1_53 {
      memref.store %cst, %alloc_51[%arg0] : memref<?xf64>
    }
    %cast_54 = memref.cast %alloc_51 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast_15, %cast_20, %cast_25, %cast_30, %cast_35, %cast_40, %cast_45, %cast_50, %cast_54, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %16 = bufferization.to_tensor %alloc_11 : memref<?xindex>
    %17 = bufferization.to_tensor %alloc_16 : memref<?xindex>
    %18 = bufferization.to_tensor %alloc_21 : memref<?xindex>
    %19 = bufferization.to_tensor %alloc_26 : memref<?xindex>
    %20 = bufferization.to_tensor %alloc_31 : memref<?xindex>
    %21 = bufferization.to_tensor %alloc_36 : memref<?xindex>
    %22 = bufferization.to_tensor %alloc_41 : memref<?xindex>
    %23 = bufferization.to_tensor %alloc_46 : memref<?xindex>
    %24 = bufferization.to_tensor %alloc_51 : memref<?xf64>
    %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
    %26 = "ta.static_index_label"(%c0_0, %12, %c1_1) : (index, index, index) -> !ta.range
    %27 = "ta.static_index_label"(%c0_2, %13, %c1_3) : (index, index, index) -> !ta.range
    %alloc_55 = memref.alloc(%12) {alignment = 32 : i64} : memref<4x?xf64>
    %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
    %alloc_56 = memref.alloc(%13) {alignment = 32 : i64} : memref<4x?xf64>
    %29 = bufferization.to_tensor %alloc_56 : memref<4x?xf64>
    %cst_57 = arith.constant 1.000000e+00 : f64
    linalg.fill ins(%cst_57 : f64) outs(%alloc_55 : memref<4x?xf64>)
    %cst_58 = arith.constant 0.000000e+00 : f64
    linalg.fill ins(%cst_58 : f64) outs(%alloc_56 : memref<4x?xf64>)
    %30 = "it.ComputeRHS"(%28, %25) {allBlocks = [["UNK", "UNK"], ["D", "D"]], allFormats = [["D", "D"], ["D", "CN"]], allPerms = [[0, 1], [1, 2]]} : (tensor<4x?xf64>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> tensor<*xf64>
    %31 = "it.ComputeLHS"(%29) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[0, 2]]} : (tensor<4x?xf64>) -> tensor<*xf64>
    %32 = "it.Compute"(%30, %31) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
    %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
    %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
    %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
    %c4_59 = arith.constant 4 : index
    %c0_60 = arith.constant 0 : index
    %c1_61 = arith.constant 1 : index
    scf.for %arg0 = %c0_60 to %c4_59 step %c1_61 {
    }
    %36 = "it.itree"(%35) : (i64) -> i64
    "ta.print"(%29) : (tensor<4x?xf64>) -> ()
    "ta.print"(%25) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @comet_sort_index(memref<*xindex>, index, index)
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4475  finished call genForOps, i = 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4394  i: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4395 %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4400 curOp is IndexTreeIndicesOp
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4401 %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4411  indices.size(): 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4442 %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4452  indices.size(): 1 tensors.size(): 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4455  Formats:D 1 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4456  Blocks:UNK 1 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4457 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4458 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4460 ---------------
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4470  call genForOps, i = 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1053  genForOps indexTreeOp
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1054 %36 = "it.itree"(%35) : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1059  genForOps ancestorsOps.size(): 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1062  ancestorsOps[0]->forOps.size(): 1, ancestorsOps->id: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1066 Tensor size: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:492 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:496  getAllocs() -  it is dense
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:501 %alloc_55 = memref.alloc(%12) {alignment = 32 : i64} : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1069 Tensors:
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1072 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:570 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:572  parent_forops.size(): 1 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:576  reset the insertion point
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:577 scf.for %arg0 = %c0_60 to %c4_59 step %c1_61 {
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:580  order: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:588  number of children: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:592 Insertion point order == 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:618  reset the insertion point
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1096  current index format: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1097  current index block: UNK
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:640  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:641 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:653  Dynamic size /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:654 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:655 %alloc_55 = memref.alloc(%12) {alignment = 32 : i64} : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:667 %12 = memref.load %alloc[%c9] : memref<13xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:680  D Loop
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:681 scf.for %arg1 = %c0_62 to %12 step %c1_63 {
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1286 finish generate loops for current index format: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4473 module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = "ta.static_index_label"(%c0, %c4, %c1) : (index, index, index) -> !ta.range
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %2 = "ta.dynamic_index_label"(%c0_2, %c1_3) : (index, index) -> !ta.range
    %alloc = memref.alloc() : memref<13xindex>
    %cast = memref.cast %alloc : memref<13xindex> to memref<*xindex>
    %c-1 = arith.constant -1 : index
    %c0_4 = arith.constant 0 : index
    %c1_5 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    call @read_input_sizes_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
    %c0_6 = arith.constant 0 : index
    %3 = memref.load %alloc[%c0_6] : memref<13xindex>
    %c1_7 = arith.constant 1 : index
    %4 = memref.load %alloc[%c1_7] : memref<13xindex>
    %c2_8 = arith.constant 2 : index
    %5 = memref.load %alloc[%c2_8] : memref<13xindex>
    %c3_9 = arith.constant 3 : index
    %6 = memref.load %alloc[%c3_9] : memref<13xindex>
    %c4_10 = arith.constant 4 : index
    %7 = memref.load %alloc[%c4_10] : memref<13xindex>
    %c5 = arith.constant 5 : index
    %8 = memref.load %alloc[%c5] : memref<13xindex>
    %c6 = arith.constant 6 : index
    %9 = memref.load %alloc[%c6] : memref<13xindex>
    %c7 = arith.constant 7 : index
    %10 = memref.load %alloc[%c7] : memref<13xindex>
    %c8 = arith.constant 8 : index
    %11 = memref.load %alloc[%c8] : memref<13xindex>
    %c9 = arith.constant 9 : index
    %12 = memref.load %alloc[%c9] : memref<13xindex>
    %c10 = arith.constant 10 : index
    %13 = memref.load %alloc[%c10] : memref<13xindex>
    %c11 = arith.constant 11 : index
    %14 = memref.load %alloc[%c11] : memref<13xindex>
    %c12 = arith.constant 12 : index
    %15 = memref.load %alloc[%c12] : memref<13xindex>
    %alloc_11 = memref.alloc(%3) : memref<?xindex>
    %c0_12 = arith.constant 0 : index
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    scf.for %arg0 = %c0_13 to %3 step %c1_14 {
      memref.store %c0_12, %alloc_11[%arg0] : memref<?xindex>
    }
    %cast_15 = memref.cast %alloc_11 : memref<?xindex> to memref<*xindex>
    %alloc_16 = memref.alloc(%4) : memref<?xindex>
    %c0_17 = arith.constant 0 : index
    %c0_18 = arith.constant 0 : index
    %c1_19 = arith.constant 1 : index
    scf.for %arg0 = %c0_18 to %4 step %c1_19 {
      memref.store %c0_17, %alloc_16[%arg0] : memref<?xindex>
    }
    %cast_20 = memref.cast %alloc_16 : memref<?xindex> to memref<*xindex>
    %alloc_21 = memref.alloc(%5) : memref<?xindex>
    %c0_22 = arith.constant 0 : index
    %c0_23 = arith.constant 0 : index
    %c1_24 = arith.constant 1 : index
    scf.for %arg0 = %c0_23 to %5 step %c1_24 {
      memref.store %c0_22, %alloc_21[%arg0] : memref<?xindex>
    }
    %cast_25 = memref.cast %alloc_21 : memref<?xindex> to memref<*xindex>
    %alloc_26 = memref.alloc(%6) : memref<?xindex>
    %c0_27 = arith.constant 0 : index
    %c0_28 = arith.constant 0 : index
    %c1_29 = arith.constant 1 : index
    scf.for %arg0 = %c0_28 to %6 step %c1_29 {
      memref.store %c0_27, %alloc_26[%arg0] : memref<?xindex>
    }
    %cast_30 = memref.cast %alloc_26 : memref<?xindex> to memref<*xindex>
    %alloc_31 = memref.alloc(%7) : memref<?xindex>
    %c0_32 = arith.constant 0 : index
    %c0_33 = arith.constant 0 : index
    %c1_34 = arith.constant 1 : index
    scf.for %arg0 = %c0_33 to %7 step %c1_34 {
      memref.store %c0_32, %alloc_31[%arg0] : memref<?xindex>
    }
    %cast_35 = memref.cast %alloc_31 : memref<?xindex> to memref<*xindex>
    %alloc_36 = memref.alloc(%8) : memref<?xindex>
    %c0_37 = arith.constant 0 : index
    %c0_38 = arith.constant 0 : index
    %c1_39 = arith.constant 1 : index
    scf.for %arg0 = %c0_38 to %8 step %c1_39 {
      memref.store %c0_37, %alloc_36[%arg0] : memref<?xindex>
    }
    %cast_40 = memref.cast %alloc_36 : memref<?xindex> to memref<*xindex>
    %alloc_41 = memref.alloc(%9) : memref<?xindex>
    %c0_42 = arith.constant 0 : index
    %c0_43 = arith.constant 0 : index
    %c1_44 = arith.constant 1 : index
    scf.for %arg0 = %c0_43 to %9 step %c1_44 {
      memref.store %c0_42, %alloc_41[%arg0] : memref<?xindex>
    }
    %cast_45 = memref.cast %alloc_41 : memref<?xindex> to memref<*xindex>
    %alloc_46 = memref.alloc(%10) : memref<?xindex>
    %c0_47 = arith.constant 0 : index
    %c0_48 = arith.constant 0 : index
    %c1_49 = arith.constant 1 : index
    scf.for %arg0 = %c0_48 to %10 step %c1_49 {
      memref.store %c0_47, %alloc_46[%arg0] : memref<?xindex>
    }
    %cast_50 = memref.cast %alloc_46 : memref<?xindex> to memref<*xindex>
    %alloc_51 = memref.alloc(%11) : memref<?xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %c0_52 = arith.constant 0 : index
    %c1_53 = arith.constant 1 : index
    scf.for %arg0 = %c0_52 to %11 step %c1_53 {
      memref.store %cst, %alloc_51[%arg0] : memref<?xf64>
    }
    %cast_54 = memref.cast %alloc_51 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast_15, %cast_20, %cast_25, %cast_30, %cast_35, %cast_40, %cast_45, %cast_50, %cast_54, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %16 = bufferization.to_tensor %alloc_11 : memref<?xindex>
    %17 = bufferization.to_tensor %alloc_16 : memref<?xindex>
    %18 = bufferization.to_tensor %alloc_21 : memref<?xindex>
    %19 = bufferization.to_tensor %alloc_26 : memref<?xindex>
    %20 = bufferization.to_tensor %alloc_31 : memref<?xindex>
    %21 = bufferization.to_tensor %alloc_36 : memref<?xindex>
    %22 = bufferization.to_tensor %alloc_41 : memref<?xindex>
    %23 = bufferization.to_tensor %alloc_46 : memref<?xindex>
    %24 = bufferization.to_tensor %alloc_51 : memref<?xf64>
    %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
    %26 = "ta.static_index_label"(%c0_0, %12, %c1_1) : (index, index, index) -> !ta.range
    %27 = "ta.static_index_label"(%c0_2, %13, %c1_3) : (index, index, index) -> !ta.range
    %alloc_55 = memref.alloc(%12) {alignment = 32 : i64} : memref<4x?xf64>
    %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
    %alloc_56 = memref.alloc(%13) {alignment = 32 : i64} : memref<4x?xf64>
    %29 = bufferization.to_tensor %alloc_56 : memref<4x?xf64>
    %cst_57 = arith.constant 1.000000e+00 : f64
    linalg.fill ins(%cst_57 : f64) outs(%alloc_55 : memref<4x?xf64>)
    %cst_58 = arith.constant 0.000000e+00 : f64
    linalg.fill ins(%cst_58 : f64) outs(%alloc_56 : memref<4x?xf64>)
    %30 = "it.ComputeRHS"(%28, %25) {allBlocks = [["UNK", "UNK"], ["D", "D"]], allFormats = [["D", "D"], ["D", "CN"]], allPerms = [[0, 1], [1, 2]]} : (tensor<4x?xf64>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> tensor<*xf64>
    %31 = "it.ComputeLHS"(%29) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[0, 2]]} : (tensor<4x?xf64>) -> tensor<*xf64>
    %32 = "it.Compute"(%30, %31) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
    %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
    %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
    %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
    %c4_59 = arith.constant 4 : index
    %c0_60 = arith.constant 0 : index
    %c1_61 = arith.constant 1 : index
    scf.for %arg0 = %c0_60 to %c4_59 step %c1_61 {
      %c0_62 = arith.constant 0 : index
      %c1_63 = arith.constant 1 : index
      scf.for %arg1 = %c0_62 to %12 step %c1_63 {
      }
    }
    %36 = "it.itree"(%35) : (i64) -> i64
    "ta.print"(%29) : (tensor<4x?xf64>) -> ()
    "ta.print"(%25) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @comet_sort_index(memref<*xindex>, index, index)
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4475  finished call genForOps, i = 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4394  i: 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4395 %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4400 curOp is IndexTreeIndicesOp
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4401 %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4411  indices.size(): 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4442 %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4452  indices.size(): 1 tensors.size(): 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4455  Formats:CN 1 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4456  Blocks:D 1 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4457 %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4458 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4460 ---------------
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4470  call genForOps, i = 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1053  genForOps indexTreeOp
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1054 %36 = "it.itree"(%35) : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1059  genForOps ancestorsOps.size(): 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1062  ancestorsOps[0]->forOps.size(): 1, ancestorsOps->id: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1062  ancestorsOps[1]->forOps.size(): 1, ancestorsOps->id: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1066 Tensor size: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:492 %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:520  getAllocs() -  it is sparse
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %16 = bufferization.to_tensor %alloc_11 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_11 = memref.alloc(%3) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %17 = bufferization.to_tensor %alloc_16 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_16 = memref.alloc(%4) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %18 = bufferization.to_tensor %alloc_21 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_21 = memref.alloc(%5) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %19 = bufferization.to_tensor %alloc_26 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_26 = memref.alloc(%6) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %20 = bufferization.to_tensor %alloc_31 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_31 = memref.alloc(%7) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %21 = bufferization.to_tensor %alloc_36 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_36 = memref.alloc(%8) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %22 = bufferization.to_tensor %alloc_41 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_41 = memref.alloc(%9) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %23 = bufferization.to_tensor %alloc_46 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_46 = memref.alloc(%10) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %24 = bufferization.to_tensor %alloc_51 : memref<?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_51 = memref.alloc(%11) : memref<?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1069 Tensors:
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1072 %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:570 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:572  parent_forops.size(): 1 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:576  reset the insertion point
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:577 scf.for %arg1 = %c0_62 to %12 step %c1_63 {
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:580  order: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:588  number of children: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:592 Insertion point order == 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:618  reset the insertion point
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1096  current index format: CN
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1097  current index block: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:849  CN Loop
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:850 scf.for %arg2 = %37 to %38 step %c1_66 {
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1243 block D for format: CN
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1286 finish generate loops for current index format: CN
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4473 module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = "ta.static_index_label"(%c0, %c4, %c1) : (index, index, index) -> !ta.range
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %2 = "ta.dynamic_index_label"(%c0_2, %c1_3) : (index, index) -> !ta.range
    %alloc = memref.alloc() : memref<13xindex>
    %cast = memref.cast %alloc : memref<13xindex> to memref<*xindex>
    %c-1 = arith.constant -1 : index
    %c0_4 = arith.constant 0 : index
    %c1_5 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    call @read_input_sizes_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
    %c0_6 = arith.constant 0 : index
    %3 = memref.load %alloc[%c0_6] : memref<13xindex>
    %c1_7 = arith.constant 1 : index
    %4 = memref.load %alloc[%c1_7] : memref<13xindex>
    %c2_8 = arith.constant 2 : index
    %5 = memref.load %alloc[%c2_8] : memref<13xindex>
    %c3_9 = arith.constant 3 : index
    %6 = memref.load %alloc[%c3_9] : memref<13xindex>
    %c4_10 = arith.constant 4 : index
    %7 = memref.load %alloc[%c4_10] : memref<13xindex>
    %c5 = arith.constant 5 : index
    %8 = memref.load %alloc[%c5] : memref<13xindex>
    %c6 = arith.constant 6 : index
    %9 = memref.load %alloc[%c6] : memref<13xindex>
    %c7 = arith.constant 7 : index
    %10 = memref.load %alloc[%c7] : memref<13xindex>
    %c8 = arith.constant 8 : index
    %11 = memref.load %alloc[%c8] : memref<13xindex>
    %c9 = arith.constant 9 : index
    %12 = memref.load %alloc[%c9] : memref<13xindex>
    %c10 = arith.constant 10 : index
    %13 = memref.load %alloc[%c10] : memref<13xindex>
    %c11 = arith.constant 11 : index
    %14 = memref.load %alloc[%c11] : memref<13xindex>
    %c12 = arith.constant 12 : index
    %15 = memref.load %alloc[%c12] : memref<13xindex>
    %alloc_11 = memref.alloc(%3) : memref<?xindex>
    %c0_12 = arith.constant 0 : index
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    scf.for %arg0 = %c0_13 to %3 step %c1_14 {
      memref.store %c0_12, %alloc_11[%arg0] : memref<?xindex>
    }
    %cast_15 = memref.cast %alloc_11 : memref<?xindex> to memref<*xindex>
    %alloc_16 = memref.alloc(%4) : memref<?xindex>
    %c0_17 = arith.constant 0 : index
    %c0_18 = arith.constant 0 : index
    %c1_19 = arith.constant 1 : index
    scf.for %arg0 = %c0_18 to %4 step %c1_19 {
      memref.store %c0_17, %alloc_16[%arg0] : memref<?xindex>
    }
    %cast_20 = memref.cast %alloc_16 : memref<?xindex> to memref<*xindex>
    %alloc_21 = memref.alloc(%5) : memref<?xindex>
    %c0_22 = arith.constant 0 : index
    %c0_23 = arith.constant 0 : index
    %c1_24 = arith.constant 1 : index
    scf.for %arg0 = %c0_23 to %5 step %c1_24 {
      memref.store %c0_22, %alloc_21[%arg0] : memref<?xindex>
    }
    %cast_25 = memref.cast %alloc_21 : memref<?xindex> to memref<*xindex>
    %alloc_26 = memref.alloc(%6) : memref<?xindex>
    %c0_27 = arith.constant 0 : index
    %c0_28 = arith.constant 0 : index
    %c1_29 = arith.constant 1 : index
    scf.for %arg0 = %c0_28 to %6 step %c1_29 {
      memref.store %c0_27, %alloc_26[%arg0] : memref<?xindex>
    }
    %cast_30 = memref.cast %alloc_26 : memref<?xindex> to memref<*xindex>
    %alloc_31 = memref.alloc(%7) : memref<?xindex>
    %c0_32 = arith.constant 0 : index
    %c0_33 = arith.constant 0 : index
    %c1_34 = arith.constant 1 : index
    scf.for %arg0 = %c0_33 to %7 step %c1_34 {
      memref.store %c0_32, %alloc_31[%arg0] : memref<?xindex>
    }
    %cast_35 = memref.cast %alloc_31 : memref<?xindex> to memref<*xindex>
    %alloc_36 = memref.alloc(%8) : memref<?xindex>
    %c0_37 = arith.constant 0 : index
    %c0_38 = arith.constant 0 : index
    %c1_39 = arith.constant 1 : index
    scf.for %arg0 = %c0_38 to %8 step %c1_39 {
      memref.store %c0_37, %alloc_36[%arg0] : memref<?xindex>
    }
    %cast_40 = memref.cast %alloc_36 : memref<?xindex> to memref<*xindex>
    %alloc_41 = memref.alloc(%9) : memref<?xindex>
    %c0_42 = arith.constant 0 : index
    %c0_43 = arith.constant 0 : index
    %c1_44 = arith.constant 1 : index
    scf.for %arg0 = %c0_43 to %9 step %c1_44 {
      memref.store %c0_42, %alloc_41[%arg0] : memref<?xindex>
    }
    %cast_45 = memref.cast %alloc_41 : memref<?xindex> to memref<*xindex>
    %alloc_46 = memref.alloc(%10) : memref<?xindex>
    %c0_47 = arith.constant 0 : index
    %c0_48 = arith.constant 0 : index
    %c1_49 = arith.constant 1 : index
    scf.for %arg0 = %c0_48 to %10 step %c1_49 {
      memref.store %c0_47, %alloc_46[%arg0] : memref<?xindex>
    }
    %cast_50 = memref.cast %alloc_46 : memref<?xindex> to memref<*xindex>
    %alloc_51 = memref.alloc(%11) : memref<?xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %c0_52 = arith.constant 0 : index
    %c1_53 = arith.constant 1 : index
    scf.for %arg0 = %c0_52 to %11 step %c1_53 {
      memref.store %cst, %alloc_51[%arg0] : memref<?xf64>
    }
    %cast_54 = memref.cast %alloc_51 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast_15, %cast_20, %cast_25, %cast_30, %cast_35, %cast_40, %cast_45, %cast_50, %cast_54, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %16 = bufferization.to_tensor %alloc_11 : memref<?xindex>
    %17 = bufferization.to_tensor %alloc_16 : memref<?xindex>
    %18 = bufferization.to_tensor %alloc_21 : memref<?xindex>
    %19 = bufferization.to_tensor %alloc_26 : memref<?xindex>
    %20 = bufferization.to_tensor %alloc_31 : memref<?xindex>
    %21 = bufferization.to_tensor %alloc_36 : memref<?xindex>
    %22 = bufferization.to_tensor %alloc_41 : memref<?xindex>
    %23 = bufferization.to_tensor %alloc_46 : memref<?xindex>
    %24 = bufferization.to_tensor %alloc_51 : memref<?xf64>
    %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
    %26 = "ta.static_index_label"(%c0_0, %12, %c1_1) : (index, index, index) -> !ta.range
    %27 = "ta.static_index_label"(%c0_2, %13, %c1_3) : (index, index, index) -> !ta.range
    %alloc_55 = memref.alloc(%12) {alignment = 32 : i64} : memref<4x?xf64>
    %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
    %alloc_56 = memref.alloc(%13) {alignment = 32 : i64} : memref<4x?xf64>
    %29 = bufferization.to_tensor %alloc_56 : memref<4x?xf64>
    %cst_57 = arith.constant 1.000000e+00 : f64
    linalg.fill ins(%cst_57 : f64) outs(%alloc_55 : memref<4x?xf64>)
    %cst_58 = arith.constant 0.000000e+00 : f64
    linalg.fill ins(%cst_58 : f64) outs(%alloc_56 : memref<4x?xf64>)
    %30 = "it.ComputeRHS"(%28, %25) {allBlocks = [["UNK", "UNK"], ["D", "D"]], allFormats = [["D", "D"], ["D", "CN"]], allPerms = [[0, 1], [1, 2]]} : (tensor<4x?xf64>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> tensor<*xf64>
    %31 = "it.ComputeLHS"(%29) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[0, 2]]} : (tensor<4x?xf64>) -> tensor<*xf64>
    %32 = "it.Compute"(%30, %31) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
    %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
    %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
    %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
    %c4_59 = arith.constant 4 : index
    %c0_60 = arith.constant 0 : index
    %c1_61 = arith.constant 1 : index
    scf.for %arg0 = %c0_60 to %c4_59 step %c1_61 {
      %c0_62 = arith.constant 0 : index
      %c1_63 = arith.constant 1 : index
      scf.for %arg1 = %c0_62 to %12 step %c1_63 {
        %c0_64 = arith.constant 0 : index
        %37 = memref.load %alloc_31[%c0_64] : memref<?xindex>
        %c1_65 = arith.constant 1 : index
        %38 = memref.load %alloc_31[%c1_65] : memref<?xindex>
        %c1_66 = arith.constant 1 : index
        scf.for %arg2 = %37 to %38 step %c1_66 {
          %39 = memref.load %alloc_36[%arg2] : memref<?xindex>
        }
      }
    }
    %36 = "it.itree"(%35) : (i64) -> i64
    "ta.print"(%29) : (tensor<4x?xf64>) -> ()
    "ta.print"(%25) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @comet_sort_index(memref<*xindex>, index, index)
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4475  finished call genForOps, i = 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4394  i: 3
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4395 %32 = "it.Compute"(%30, %31) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4492  call genCmptOps, i = 3
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3695  calling genCmptOps
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3697  
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3699  Current IndexTreeComputeOp:/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3700 %32 = "it.Compute"(%30, %31) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3703  comp_worksp_opt (bool: true is compressed): 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3712  ancestorsOps.size(): 3
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3715  ancestorsOps[i]->id:2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3715  ancestorsOps[i]->id:1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3715  ancestorsOps[i]->id:0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2369  ancestorsOps[0]->forOps.size(): 1, ancestorsOps->id: 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2376  j: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2378 AccessIdx: %39 = memref.load %alloc_36[%arg2] : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2369  ancestorsOps[1]->forOps.size(): 1, ancestorsOps->id: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2376  j: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2378 AccessIdx: <block argument> of type 'index' at index: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2369  ancestorsOps[2]->forOps.size(): 1, ancestorsOps->id: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2376  j: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2378 AccessIdx: <block argument> of type 'index' at index: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2383  nested_forops.size(): 3
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2386  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2387 %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2386  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2387 %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2386  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2387 %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3728  nested_forops_indices.size(): 3
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3734  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3735 ^bb0(%arg2: index):
  %39 = memref.load %alloc_36[%arg2] : memref<?xindex>
  scf.yield
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3736  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3737 scf.yield
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3740 scf.for %arg2 = %37 to %38 step %c1_66 {
  %39 = memref.load %alloc_36[%arg2] : memref<?xindex>
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2420 %32 = "it.Compute"(%30, %31) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2423  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2424 %30 = "it.ComputeRHS"(%28, %25) {allBlocks = [["UNK", "UNK"], ["D", "D"]], allFormats = [["D", "D"], ["D", "CN"]], allPerms = [[0, 1], [1, 2]]} : (tensor<4x?xf64>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> tensor<*xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2427  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2428 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2427  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2428 %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2436  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2437 %29 = bufferization.to_tensor %alloc_56 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:492 %29 = bufferization.to_tensor %alloc_56 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:496  getAllocs() -  it is dense
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:501 %alloc_56 = memref.alloc(%13) {alignment = 32 : i64} : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2443  tensors_lhs_Allocs.size(): 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:492 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:496  getAllocs() -  it is dense
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:501 %alloc_55 = memref.alloc(%12) {alignment = 32 : i64} : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:492 %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:520  getAllocs() -  it is sparse
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %16 = bufferization.to_tensor %alloc_11 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_11 = memref.alloc(%3) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %17 = bufferization.to_tensor %alloc_16 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_16 = memref.alloc(%4) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %18 = bufferization.to_tensor %alloc_21 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_21 = memref.alloc(%5) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %19 = bufferization.to_tensor %alloc_26 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_26 = memref.alloc(%6) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %20 = bufferization.to_tensor %alloc_31 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_31 = memref.alloc(%7) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %21 = bufferization.to_tensor %alloc_36 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_36 = memref.alloc(%8) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %22 = bufferization.to_tensor %alloc_41 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_41 = memref.alloc(%9) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %23 = bufferization.to_tensor %alloc_46 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_46 = memref.alloc(%10) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %24 = bufferization.to_tensor %alloc_51 : memref<?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_51 = memref.alloc(%11) : memref<?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2445  tensors_rhs_Allocs.size(): 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2462  allPerms: 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2465  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2468 0 /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2468 1 /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2470 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2465  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2468 1 /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2468 2 /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2470 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2465  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2468 0 /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2468 2 /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2470 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2474  allFormats: 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2477  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2480 D /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2480 D /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2482 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2477  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2480 D /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2480 CN /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2482 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2477  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2480 D /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2480 D /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2482 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2486  allBlocks: 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2489  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2492 UNK /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2492 UNK /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2494 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2489  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2492 D /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2492 D /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2494 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2489  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2492 UNK /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2492 UNK /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2494 
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2497  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2498 %32 = "it.Compute"(%30, %31) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2505  allPerms.size(): 3
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2510  allPerms_rhs.size(): 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2514  allPerms_lhs.size(): 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2526  main_tensors_rhs.size(): 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2538  main_tensors_lhs.size(): 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2542  main_tensors_all.size(): 3
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3771  main_tensor_nums: 3
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:492 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:496  getAllocs() -  it is dense
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:501 %alloc_55 = memref.alloc(%12) {alignment = 32 : i64} : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:492 %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:520  getAllocs() -  it is sparse
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %16 = bufferization.to_tensor %alloc_11 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_11 = memref.alloc(%3) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %17 = bufferization.to_tensor %alloc_16 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_16 = memref.alloc(%4) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %18 = bufferization.to_tensor %alloc_21 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_21 = memref.alloc(%5) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %19 = bufferization.to_tensor %alloc_26 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_26 = memref.alloc(%6) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %20 = bufferization.to_tensor %alloc_31 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_31 = memref.alloc(%7) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %21 = bufferization.to_tensor %alloc_36 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_36 = memref.alloc(%8) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %22 = bufferization.to_tensor %alloc_41 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_41 = memref.alloc(%9) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %23 = bufferization.to_tensor %alloc_46 : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_46 = memref.alloc(%10) : memref<?xindex>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:525 %24 = bufferization.to_tensor %alloc_51 : memref<?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:529 %alloc_51 = memref.alloc(%11) : memref<?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:492 %29 = bufferization.to_tensor %alloc_56 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:496  getAllocs() -  it is dense
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:501 %alloc_56 = memref.alloc(%13) {alignment = 32 : i64} : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3774  main_tensors_all_Allocs.size(): 3
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2571  index_loc 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2572  Perm: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2573  Format: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2574  Block: UNK
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2571  index_loc 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2572  Perm: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2573  Format: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2574  Block: UNK
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2571  index_loc 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2572  Perm: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2573  Format: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2574  Block: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2571  index_loc 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2572  Perm: 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2573  Format: CN
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2574  Block: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2571  index_loc 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2572  Perm: 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2573  Format: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2574  Block: UNK
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2571  index_loc 0
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2572  Perm: 2
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2573  Format: D
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2574  Block: UNK
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2587  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2588 %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2587  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2588 %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2587  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2588 %29 = bufferization.to_tensor %alloc_56 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:3843 %29 = bufferization.to_tensor %alloc_56 : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4045 No masking codegen...
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1853  isElementwise:0 isMixedMode: 1
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1960  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1961 %40 = memref.load %alloc_55[%arg0, %arg1] : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1960  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1961 %41 = memref.load %alloc_51[%arg2] : memref<?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1960  /home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1961 %42 = memref.load %alloc_56[%arg0, %39] : memref<4x?xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:1963  allLoads.size(): 3
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:2240 calculate elementWise operation and reduction for general dense or mix mode computation (which has dense output)
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4497 module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = "ta.static_index_label"(%c0, %c4, %c1) : (index, index, index) -> !ta.range
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %2 = "ta.dynamic_index_label"(%c0_2, %c1_3) : (index, index) -> !ta.range
    %alloc = memref.alloc() : memref<13xindex>
    %cast = memref.cast %alloc : memref<13xindex> to memref<*xindex>
    %c-1 = arith.constant -1 : index
    %c0_4 = arith.constant 0 : index
    %c1_5 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    call @read_input_sizes_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
    %c0_6 = arith.constant 0 : index
    %3 = memref.load %alloc[%c0_6] : memref<13xindex>
    %c1_7 = arith.constant 1 : index
    %4 = memref.load %alloc[%c1_7] : memref<13xindex>
    %c2_8 = arith.constant 2 : index
    %5 = memref.load %alloc[%c2_8] : memref<13xindex>
    %c3_9 = arith.constant 3 : index
    %6 = memref.load %alloc[%c3_9] : memref<13xindex>
    %c4_10 = arith.constant 4 : index
    %7 = memref.load %alloc[%c4_10] : memref<13xindex>
    %c5 = arith.constant 5 : index
    %8 = memref.load %alloc[%c5] : memref<13xindex>
    %c6 = arith.constant 6 : index
    %9 = memref.load %alloc[%c6] : memref<13xindex>
    %c7 = arith.constant 7 : index
    %10 = memref.load %alloc[%c7] : memref<13xindex>
    %c8 = arith.constant 8 : index
    %11 = memref.load %alloc[%c8] : memref<13xindex>
    %c9 = arith.constant 9 : index
    %12 = memref.load %alloc[%c9] : memref<13xindex>
    %c10 = arith.constant 10 : index
    %13 = memref.load %alloc[%c10] : memref<13xindex>
    %c11 = arith.constant 11 : index
    %14 = memref.load %alloc[%c11] : memref<13xindex>
    %c12 = arith.constant 12 : index
    %15 = memref.load %alloc[%c12] : memref<13xindex>
    %alloc_11 = memref.alloc(%3) : memref<?xindex>
    %c0_12 = arith.constant 0 : index
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    scf.for %arg0 = %c0_13 to %3 step %c1_14 {
      memref.store %c0_12, %alloc_11[%arg0] : memref<?xindex>
    }
    %cast_15 = memref.cast %alloc_11 : memref<?xindex> to memref<*xindex>
    %alloc_16 = memref.alloc(%4) : memref<?xindex>
    %c0_17 = arith.constant 0 : index
    %c0_18 = arith.constant 0 : index
    %c1_19 = arith.constant 1 : index
    scf.for %arg0 = %c0_18 to %4 step %c1_19 {
      memref.store %c0_17, %alloc_16[%arg0] : memref<?xindex>
    }
    %cast_20 = memref.cast %alloc_16 : memref<?xindex> to memref<*xindex>
    %alloc_21 = memref.alloc(%5) : memref<?xindex>
    %c0_22 = arith.constant 0 : index
    %c0_23 = arith.constant 0 : index
    %c1_24 = arith.constant 1 : index
    scf.for %arg0 = %c0_23 to %5 step %c1_24 {
      memref.store %c0_22, %alloc_21[%arg0] : memref<?xindex>
    }
    %cast_25 = memref.cast %alloc_21 : memref<?xindex> to memref<*xindex>
    %alloc_26 = memref.alloc(%6) : memref<?xindex>
    %c0_27 = arith.constant 0 : index
    %c0_28 = arith.constant 0 : index
    %c1_29 = arith.constant 1 : index
    scf.for %arg0 = %c0_28 to %6 step %c1_29 {
      memref.store %c0_27, %alloc_26[%arg0] : memref<?xindex>
    }
    %cast_30 = memref.cast %alloc_26 : memref<?xindex> to memref<*xindex>
    %alloc_31 = memref.alloc(%7) : memref<?xindex>
    %c0_32 = arith.constant 0 : index
    %c0_33 = arith.constant 0 : index
    %c1_34 = arith.constant 1 : index
    scf.for %arg0 = %c0_33 to %7 step %c1_34 {
      memref.store %c0_32, %alloc_31[%arg0] : memref<?xindex>
    }
    %cast_35 = memref.cast %alloc_31 : memref<?xindex> to memref<*xindex>
    %alloc_36 = memref.alloc(%8) : memref<?xindex>
    %c0_37 = arith.constant 0 : index
    %c0_38 = arith.constant 0 : index
    %c1_39 = arith.constant 1 : index
    scf.for %arg0 = %c0_38 to %8 step %c1_39 {
      memref.store %c0_37, %alloc_36[%arg0] : memref<?xindex>
    }
    %cast_40 = memref.cast %alloc_36 : memref<?xindex> to memref<*xindex>
    %alloc_41 = memref.alloc(%9) : memref<?xindex>
    %c0_42 = arith.constant 0 : index
    %c0_43 = arith.constant 0 : index
    %c1_44 = arith.constant 1 : index
    scf.for %arg0 = %c0_43 to %9 step %c1_44 {
      memref.store %c0_42, %alloc_41[%arg0] : memref<?xindex>
    }
    %cast_45 = memref.cast %alloc_41 : memref<?xindex> to memref<*xindex>
    %alloc_46 = memref.alloc(%10) : memref<?xindex>
    %c0_47 = arith.constant 0 : index
    %c0_48 = arith.constant 0 : index
    %c1_49 = arith.constant 1 : index
    scf.for %arg0 = %c0_48 to %10 step %c1_49 {
      memref.store %c0_47, %alloc_46[%arg0] : memref<?xindex>
    }
    %cast_50 = memref.cast %alloc_46 : memref<?xindex> to memref<*xindex>
    %alloc_51 = memref.alloc(%11) : memref<?xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %c0_52 = arith.constant 0 : index
    %c1_53 = arith.constant 1 : index
    scf.for %arg0 = %c0_52 to %11 step %c1_53 {
      memref.store %cst, %alloc_51[%arg0] : memref<?xf64>
    }
    %cast_54 = memref.cast %alloc_51 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast_15, %cast_20, %cast_25, %cast_30, %cast_35, %cast_40, %cast_45, %cast_50, %cast_54, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %16 = bufferization.to_tensor %alloc_11 : memref<?xindex>
    %17 = bufferization.to_tensor %alloc_16 : memref<?xindex>
    %18 = bufferization.to_tensor %alloc_21 : memref<?xindex>
    %19 = bufferization.to_tensor %alloc_26 : memref<?xindex>
    %20 = bufferization.to_tensor %alloc_31 : memref<?xindex>
    %21 = bufferization.to_tensor %alloc_36 : memref<?xindex>
    %22 = bufferization.to_tensor %alloc_41 : memref<?xindex>
    %23 = bufferization.to_tensor %alloc_46 : memref<?xindex>
    %24 = bufferization.to_tensor %alloc_51 : memref<?xf64>
    %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
    %26 = "ta.static_index_label"(%c0_0, %12, %c1_1) : (index, index, index) -> !ta.range
    %27 = "ta.static_index_label"(%c0_2, %13, %c1_3) : (index, index, index) -> !ta.range
    %alloc_55 = memref.alloc(%12) {alignment = 32 : i64} : memref<4x?xf64>
    %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
    %alloc_56 = memref.alloc(%13) {alignment = 32 : i64} : memref<4x?xf64>
    %29 = bufferization.to_tensor %alloc_56 : memref<4x?xf64>
    %cst_57 = arith.constant 1.000000e+00 : f64
    linalg.fill ins(%cst_57 : f64) outs(%alloc_55 : memref<4x?xf64>)
    %cst_58 = arith.constant 0.000000e+00 : f64
    linalg.fill ins(%cst_58 : f64) outs(%alloc_56 : memref<4x?xf64>)
    %30 = "it.ComputeRHS"(%28, %25) {allBlocks = [["UNK", "UNK"], ["D", "D"]], allFormats = [["D", "D"], ["D", "CN"]], allPerms = [[0, 1], [1, 2]]} : (tensor<4x?xf64>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> tensor<*xf64>
    %31 = "it.ComputeLHS"(%29) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[0, 2]]} : (tensor<4x?xf64>) -> tensor<*xf64>
    %32 = "it.Compute"(%30, %31) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
    %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
    %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
    %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
    %c4_59 = arith.constant 4 : index
    %c0_60 = arith.constant 0 : index
    %c1_61 = arith.constant 1 : index
    scf.for %arg0 = %c0_60 to %c4_59 step %c1_61 {
      %c0_62 = arith.constant 0 : index
      %c1_63 = arith.constant 1 : index
      scf.for %arg1 = %c0_62 to %12 step %c1_63 {
        %c0_64 = arith.constant 0 : index
        %37 = memref.load %alloc_31[%c0_64] : memref<?xindex>
        %c1_65 = arith.constant 1 : index
        %38 = memref.load %alloc_31[%c1_65] : memref<?xindex>
        %c1_66 = arith.constant 1 : index
        scf.for %arg2 = %37 to %38 step %c1_66 {
          %39 = memref.load %alloc_36[%arg2] : memref<?xindex>
          %cst_67 = arith.constant 0.000000e+00 : f64
          %40 = memref.load %alloc_55[%arg0, %arg1] : memref<4x?xf64>
          %41 = memref.load %alloc_51[%arg2] : memref<?xf64>
          %42 = memref.load %alloc_56[%arg0, %39] : memref<4x?xf64>
          %43 = arith.mulf %40, %41 : f64
          %44 = arith.addf %42, %43 : f64
          memref.store %44, %alloc_56[%arg0, %39] : memref<4x?xf64>
        }
      }
    }
    %36 = "it.itree"(%35) : (i64) -> i64
    "ta.print"(%29) : (tensor<4x?xf64>) -> ()
    "ta.print"(%25) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @comet_sort_index(memref<*xindex>, index, index)
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4499  finished call genCmptOps, i = 3
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4504 End of doLoweringIndexTreeToSCF()
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4505 module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = "ta.static_index_label"(%c0, %c4, %c1) : (index, index, index) -> !ta.range
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %1 = "ta.dynamic_index_label"(%c0_0, %c1_1) : (index, index) -> !ta.range
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %2 = "ta.dynamic_index_label"(%c0_2, %c1_3) : (index, index) -> !ta.range
    %alloc = memref.alloc() : memref<13xindex>
    %cast = memref.cast %alloc : memref<13xindex> to memref<*xindex>
    %c-1 = arith.constant -1 : index
    %c0_4 = arith.constant 0 : index
    %c1_5 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    call @read_input_sizes_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, i32) -> ()
    %c0_6 = arith.constant 0 : index
    %3 = memref.load %alloc[%c0_6] : memref<13xindex>
    %c1_7 = arith.constant 1 : index
    %4 = memref.load %alloc[%c1_7] : memref<13xindex>
    %c2_8 = arith.constant 2 : index
    %5 = memref.load %alloc[%c2_8] : memref<13xindex>
    %c3_9 = arith.constant 3 : index
    %6 = memref.load %alloc[%c3_9] : memref<13xindex>
    %c4_10 = arith.constant 4 : index
    %7 = memref.load %alloc[%c4_10] : memref<13xindex>
    %c5 = arith.constant 5 : index
    %8 = memref.load %alloc[%c5] : memref<13xindex>
    %c6 = arith.constant 6 : index
    %9 = memref.load %alloc[%c6] : memref<13xindex>
    %c7 = arith.constant 7 : index
    %10 = memref.load %alloc[%c7] : memref<13xindex>
    %c8 = arith.constant 8 : index
    %11 = memref.load %alloc[%c8] : memref<13xindex>
    %c9 = arith.constant 9 : index
    %12 = memref.load %alloc[%c9] : memref<13xindex>
    %c10 = arith.constant 10 : index
    %13 = memref.load %alloc[%c10] : memref<13xindex>
    %c11 = arith.constant 11 : index
    %14 = memref.load %alloc[%c11] : memref<13xindex>
    %c12 = arith.constant 12 : index
    %15 = memref.load %alloc[%c12] : memref<13xindex>
    %alloc_11 = memref.alloc(%3) : memref<?xindex>
    %c0_12 = arith.constant 0 : index
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    scf.for %arg0 = %c0_13 to %3 step %c1_14 {
      memref.store %c0_12, %alloc_11[%arg0] : memref<?xindex>
    }
    %cast_15 = memref.cast %alloc_11 : memref<?xindex> to memref<*xindex>
    %alloc_16 = memref.alloc(%4) : memref<?xindex>
    %c0_17 = arith.constant 0 : index
    %c0_18 = arith.constant 0 : index
    %c1_19 = arith.constant 1 : index
    scf.for %arg0 = %c0_18 to %4 step %c1_19 {
      memref.store %c0_17, %alloc_16[%arg0] : memref<?xindex>
    }
    %cast_20 = memref.cast %alloc_16 : memref<?xindex> to memref<*xindex>
    %alloc_21 = memref.alloc(%5) : memref<?xindex>
    %c0_22 = arith.constant 0 : index
    %c0_23 = arith.constant 0 : index
    %c1_24 = arith.constant 1 : index
    scf.for %arg0 = %c0_23 to %5 step %c1_24 {
      memref.store %c0_22, %alloc_21[%arg0] : memref<?xindex>
    }
    %cast_25 = memref.cast %alloc_21 : memref<?xindex> to memref<*xindex>
    %alloc_26 = memref.alloc(%6) : memref<?xindex>
    %c0_27 = arith.constant 0 : index
    %c0_28 = arith.constant 0 : index
    %c1_29 = arith.constant 1 : index
    scf.for %arg0 = %c0_28 to %6 step %c1_29 {
      memref.store %c0_27, %alloc_26[%arg0] : memref<?xindex>
    }
    %cast_30 = memref.cast %alloc_26 : memref<?xindex> to memref<*xindex>
    %alloc_31 = memref.alloc(%7) : memref<?xindex>
    %c0_32 = arith.constant 0 : index
    %c0_33 = arith.constant 0 : index
    %c1_34 = arith.constant 1 : index
    scf.for %arg0 = %c0_33 to %7 step %c1_34 {
      memref.store %c0_32, %alloc_31[%arg0] : memref<?xindex>
    }
    %cast_35 = memref.cast %alloc_31 : memref<?xindex> to memref<*xindex>
    %alloc_36 = memref.alloc(%8) : memref<?xindex>
    %c0_37 = arith.constant 0 : index
    %c0_38 = arith.constant 0 : index
    %c1_39 = arith.constant 1 : index
    scf.for %arg0 = %c0_38 to %8 step %c1_39 {
      memref.store %c0_37, %alloc_36[%arg0] : memref<?xindex>
    }
    %cast_40 = memref.cast %alloc_36 : memref<?xindex> to memref<*xindex>
    %alloc_41 = memref.alloc(%9) : memref<?xindex>
    %c0_42 = arith.constant 0 : index
    %c0_43 = arith.constant 0 : index
    %c1_44 = arith.constant 1 : index
    scf.for %arg0 = %c0_43 to %9 step %c1_44 {
      memref.store %c0_42, %alloc_41[%arg0] : memref<?xindex>
    }
    %cast_45 = memref.cast %alloc_41 : memref<?xindex> to memref<*xindex>
    %alloc_46 = memref.alloc(%10) : memref<?xindex>
    %c0_47 = arith.constant 0 : index
    %c0_48 = arith.constant 0 : index
    %c1_49 = arith.constant 1 : index
    scf.for %arg0 = %c0_48 to %10 step %c1_49 {
      memref.store %c0_47, %alloc_46[%arg0] : memref<?xindex>
    }
    %cast_50 = memref.cast %alloc_46 : memref<?xindex> to memref<*xindex>
    %alloc_51 = memref.alloc(%11) : memref<?xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %c0_52 = arith.constant 0 : index
    %c1_53 = arith.constant 1 : index
    scf.for %arg0 = %c0_52 to %11 step %c1_53 {
      memref.store %cst, %alloc_51[%arg0] : memref<?xf64>
    }
    %cast_54 = memref.cast %alloc_51 : memref<?xf64> to memref<*xf64>
    call @read_input_2D_f64(%c0_i32, %c0_4, %c0_4, %c2, %c0_4, %cast_15, %cast_20, %cast_25, %cast_30, %cast_35, %cast_40, %cast_45, %cast_50, %cast_54, %c1_i32) {filename = "SPARSE_FILE_NAME0"} : (i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32) -> ()
    %16 = bufferization.to_tensor %alloc_11 : memref<?xindex>
    %17 = bufferization.to_tensor %alloc_16 : memref<?xindex>
    %18 = bufferization.to_tensor %alloc_21 : memref<?xindex>
    %19 = bufferization.to_tensor %alloc_26 : memref<?xindex>
    %20 = bufferization.to_tensor %alloc_31 : memref<?xindex>
    %21 = bufferization.to_tensor %alloc_36 : memref<?xindex>
    %22 = bufferization.to_tensor %alloc_41 : memref<?xindex>
    %23 = bufferization.to_tensor %alloc_46 : memref<?xindex>
    %24 = bufferization.to_tensor %alloc_51 : memref<?xf64>
    %25 = ta.sptensor_construct(%16, %17, %18, %19, %20, %21, %22, %23, %24, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) {tensor_rank = 2 : i32} : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index) -> (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>)
    %26 = "ta.static_index_label"(%c0_0, %12, %c1_1) : (index, index, index) -> !ta.range
    %27 = "ta.static_index_label"(%c0_2, %13, %c1_3) : (index, index, index) -> !ta.range
    %alloc_55 = memref.alloc(%12) {alignment = 32 : i64} : memref<4x?xf64>
    %28 = bufferization.to_tensor %alloc_55 : memref<4x?xf64>
    %alloc_56 = memref.alloc(%13) {alignment = 32 : i64} : memref<4x?xf64>
    %29 = bufferization.to_tensor %alloc_56 : memref<4x?xf64>
    %cst_57 = arith.constant 1.000000e+00 : f64
    linalg.fill ins(%cst_57 : f64) outs(%alloc_55 : memref<4x?xf64>)
    %cst_58 = arith.constant 0.000000e+00 : f64
    linalg.fill ins(%cst_58 : f64) outs(%alloc_56 : memref<4x?xf64>)
    %30 = "it.ComputeRHS"(%28, %25) {allBlocks = [["UNK", "UNK"], ["D", "D"]], allFormats = [["D", "D"], ["D", "CN"]], allPerms = [[0, 1], [1, 2]]} : (tensor<4x?xf64>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> tensor<*xf64>
    %31 = "it.ComputeLHS"(%29) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[0, 2]]} : (tensor<4x?xf64>) -> tensor<*xf64>
    %32 = "it.Compute"(%30, %31) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
    %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
    %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
    %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
    %c4_59 = arith.constant 4 : index
    %c0_60 = arith.constant 0 : index
    %c1_61 = arith.constant 1 : index
    scf.for %arg0 = %c0_60 to %c4_59 step %c1_61 {
      %c0_62 = arith.constant 0 : index
      %c1_63 = arith.constant 1 : index
      scf.for %arg1 = %c0_62 to %12 step %c1_63 {
        %c0_64 = arith.constant 0 : index
        %37 = memref.load %alloc_31[%c0_64] : memref<?xindex>
        %c1_65 = arith.constant 1 : index
        %38 = memref.load %alloc_31[%c1_65] : memref<?xindex>
        %c1_66 = arith.constant 1 : index
        scf.for %arg2 = %37 to %38 step %c1_66 {
          %39 = memref.load %alloc_36[%arg2] : memref<?xindex>
          %cst_67 = arith.constant 0.000000e+00 : f64
          %40 = memref.load %alloc_55[%arg0, %arg1] : memref<4x?xf64>
          %41 = memref.load %alloc_51[%arg2] : memref<?xf64>
          %42 = memref.load %alloc_56[%arg0, %39] : memref<4x?xf64>
          %43 = arith.mulf %40, %41 : f64
          %44 = arith.addf %42, %43 : f64
          memref.store %44, %alloc_56[%arg0, %39] : memref<4x?xf64>
        }
      }
    }
    %36 = "it.itree"(%35) : (i64) -> i64
    "ta.print"(%29) : (tensor<4x?xf64>) -> ()
    "ta.print"(%25) : (!ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> ()
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @comet_sort_index(memref<*xindex>, index, index)
}
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4508 Cleaning up IndexTree Operations
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4509 %36 = "it.itree"(%35) : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4521 %35 = "it.Indices"(%34) {indices = [0]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4521 %34 = "it.Indices"(%33) {indices = [1]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4521 %33 = "it.Indices"(%32) {indices = [2]} : (i64) -> i64
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4516 %30 = "it.ComputeRHS"(%28, %25) {allBlocks = [["UNK", "UNK"], ["D", "D"]], allFormats = [["D", "D"], ["D", "CN"]], allPerms = [[0, 1], [1, 2]]} : (tensor<4x?xf64>, !ta.sptensor<tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xindex>, tensor<?xf64>, index, index, index, index, index, index, index, index, index, index, index>) -> tensor<*xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4517 %31 = "it.ComputeLHS"(%29) {allBlocks = [["UNK", "UNK"]], allFormats = [["D", "D"]], allPerms = [[0, 2]]} : (tensor<4x?xf64>) -> tensor<*xf64>
/home/patrick/Work/PNNL/COMET/lib/Conversion/IndexTreeToSCF/IndexTreeToSCF.cpp:4521 %32 = "it.Compute"(%30, %31) {MaskType = "none", comp_worksp_opt = false, semiring = "plusxy_times"} : (tensor<*xf64>, tensor<*xf64>) -> i64
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
    scf.for %arg0 = %c0 to %c4 step %c1 {
      scf.for %arg1 = %c0 to %9 step %c1 {
        %11 = memref.load %alloc_9[%c0] : memref<?xindex>
        %12 = memref.load %alloc_9[%c1] : memref<?xindex>
        scf.for %arg2 = %11 to %12 step %c1 {
          %13 = memref.load %alloc_11[%arg2] : memref<?xindex>
          %14 = memref.load %alloc_19[%arg0, %arg1] : memref<4x?xf64>
          %15 = memref.load %alloc_17[%arg2] : memref<?xf64>
          %16 = memref.load %alloc_20[%arg0, %13] : memref<4x?xf64>
          %17 = arith.mulf %14, %15 : f64
          %18 = arith.addf %16, %17 : f64
          memref.store %18, %alloc_20[%arg0, %13] : memref<4x?xf64>
        }
      }
    }
    %cast_21 = memref.cast %alloc_20 : memref<4x?xf64> to memref<*xf64>
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
    return
  }
  func.func private @read_input_2D_f64(i32, index, index, index, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xf64>, i32)
  func.func private @read_input_sizes_2D_f64(i32, index, index, index, index, memref<*xindex>, i32)
  func.func private @comet_sort_index(memref<*xindex>, index, index)
  func.func private @comet_print_memref_f64(memref<*xf64>)
  func.func private @comet_print_memref_i64(memref<*xindex>)
}
